"""Token-relation, reliability-guided TC-TD V3 universal patch optimizer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

EPS = 1e-8
LOCAL_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
    (-2, 0), (0, -2), (0, 2), (2, 0),
)


@dataclass(frozen=True)
class SampleTransform:
    crop_top: int
    crop_left: int
    crop_height: int
    crop_width: int
    brightness: float
    contrast: float
    patch_height: int
    patch_width: int
    patch_top: int
    patch_left: int
    patch_brightness: float


def local_token_relations(
    features: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local relation logits, probabilities, and a valid-neighbor mask."""
    if features.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, dim], got {tuple(features.shape)}")
    batch, tokens, _ = features.shape
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError(f"Token count {tokens} is not a square spatial grid")

    normalized = F.normalize(features.float(), dim=-1, eps=EPS).reshape(batch, side, side, -1)
    rows = torch.arange(side, device=features.device).view(side, 1)
    cols = torch.arange(side, device=features.device).view(1, side)
    similarities = []
    masks = []
    for row_offset, col_offset in LOCAL_OFFSETS:
        shifted = torch.roll(normalized, shifts=(-row_offset, -col_offset), dims=(1, 2))
        similarities.append(torch.sum(normalized * shifted, dim=-1))
        valid = (
            (rows + row_offset >= 0)
            & (rows + row_offset < side)
            & (cols + col_offset >= 0)
            & (cols + col_offset < side)
        )
        masks.append(valid)

    logits = torch.stack(similarities, dim=-1).reshape(batch, tokens, -1)
    valid_mask = torch.stack(masks, dim=-1).reshape(1, tokens, -1)
    masked_logits = (logits / temperature).masked_fill(
        ~valid_mask, torch.finfo(logits.dtype).min
    )
    probabilities = F.softmax(masked_logits, dim=-1)
    return logits.masked_fill(~valid_mask, 0.0), probabilities, valid_mask


def jensen_shannon_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return a stable Jensen-Shannon divergence for each batch/token row."""
    if p.shape != q.shape:
        raise ValueError(f"Distribution shape mismatch: {tuple(p.shape)} vs {tuple(q.shape)}")
    p = p.float().clamp_min(EPS)
    q = q.float().clamp_min(EPS)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * (torch.log(p) - torch.log(midpoint)), dim=-1)
        + torch.sum(q * (torch.log(q) - torch.log(midpoint)), dim=-1)
    )


def relation_direction(delta: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Flatten and normalize per-sample relation changes, excluding invalid edges."""
    masked = delta * valid_mask.to(delta.dtype)
    return F.normalize(masked.flatten(start_dim=1), dim=-1, eps=EPS)


def reliability_weighted_gradient(
    gradients: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float], float, float]:
    """Aggregate unit gradients using their positive agreement with all peers."""
    if not gradients:
        raise ValueError("At least one gradient is required")
    flat = torch.stack(
        [F.normalize(gradient.float().flatten(), dim=0, eps=EPS) for gradient in gradients]
    )
    if len(gradients) == 1:
        weights = torch.ones(1, device=flat.device)
        cosine_mean = cosine_min = 1.0
    else:
        cosine = flat @ flat.transpose(0, 1)
        off_diagonal = ~torch.eye(len(gradients), dtype=torch.bool, device=flat.device)
        positive_support = cosine.clamp_min(0.0).masked_fill(~off_diagonal, 0.0)
        reliability = positive_support.sum(dim=1) / (len(gradients) - 1)
        weights = reliability / reliability.sum() if reliability.sum() > EPS else torch.full_like(reliability, 1 / len(gradients))
        pairwise = cosine[off_diagonal]
        cosine_mean = pairwise.mean().item()
        cosine_min = pairwise.min().item()
    combined = torch.sum(flat * weights.unsqueeze(1), dim=0).reshape_as(gradients[0])
    return combined.to(gradients[0].dtype), weights.tolist(), cosine_mean, cosine_min


def scale_and_project_to_anchor(
    auxiliary: torch.Tensor,
    anchor: torch.Tensor,
) -> tuple[torch.Tensor, float, float, bool]:
    """Rescale the auxiliary gradient to the anchor norm and drop any component that opposes it."""
    anchor_float = anchor.float()
    auxiliary_float = auxiliary.float()
    auxiliary_norm = torch.linalg.vector_norm(auxiliary_float)
    anchor_norm = torch.linalg.vector_norm(anchor_float)
    if auxiliary_norm <= EPS or anchor_norm <= EPS:
        return torch.zeros_like(auxiliary), 0.0, 0.0, False
    scaled = auxiliary * (anchor_norm / auxiliary_norm).to(auxiliary.dtype)
    dot_before = torch.sum(scaled.float() * anchor_float)
    projected = scaled
    was_projected = bool(dot_before.item() < 0.0)
    if was_projected:
        projected = scaled - (dot_before / (anchor_norm.square() + EPS)).to(scaled.dtype) * anchor
    dot_after = torch.sum(projected.float() * anchor_float)
    return projected, dot_before.item(), dot_after.item(), was_projected


class TokenRelationTCTDV3Attacker:
    def __init__(self, cfg, device_id: int | torch.device):
        self.cfg = cfg
        self.device_id = device_id
        self.max_steps = cfg.max_steps
        self.save_steps = cfg.save_steps
        self.step_size = cfg.step_size
        self.alpha = cfg.alpha
        self.image_size = cfg.image_size
        self.perturbation_ratio = cfg.perturbation_ratio
        self.num_views = cfg.num_views
        self.relation_temperature = cfg.relation_temperature
        self.ema_decay = 0.9
        self.ema_warmup = 100
        self.ema_step = 0
        self.ema_patch = torch.tensor(0.0)
        self.ema_align = torch.tensor(0.0)
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        backend = getattr(cfg, "embedding_backend", "openvla")
        if backend == "openvla":
            from utils.openvla import get_img_embedding, get_lang_embedding, image_transform
        elif backend == "openvla_oft":
            from utils.openvla_oft import get_img_embedding, get_lang_embedding, image_transform
        else:
            raise ValueError(f"Unsupported V3 PyTorch backend: {backend}")
        self.get_img_embedding = get_img_embedding
        self.get_lang_embedding = get_lang_embedding
        self.image_transform = image_transform

        self.output_dir = Path(f"{cfg.save_path}-{cfg.perturbation_ratio}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    @staticmethod
    def convert_to_tensor(images) -> torch.Tensor:
        return torch.stack([transforms.ToTensor()(image) for image in images])

    @staticmethod
    def _rand_uniform(generator: torch.Generator, low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=generator).item()

    def preprocess(self, images: torch.Tensor, image_processor) -> torch.Tensor:
        return torch.stack([self.image_transform(image, image_processor) for image in images])

    @staticmethod
    def branch_features(vla, preprocessed_images: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = vla.vision_backbone
        if not getattr(backbone, "use_fused_vision_backbone", False):
            raise ValueError("TC-TD V3 requires the fused DINOv2+SigLIP OpenVLA backbone")
        raw = backbone(preprocessed_images.to(torch.bfloat16).to(vla.device))
        dino_dim = backbone.featurizer.embed_dim
        siglip_dim = backbone.fused_featurizer.embed_dim
        if raw.ndim != 3 or raw.shape[-1] != dino_dim + siglip_dim:
            raise ValueError(f"Unexpected fused feature shape: {tuple(raw.shape)}")
        dino, siglip = torch.split(raw, [dino_dim, siglip_dim], dim=-1)
        return {"dino": dino, "siglip": siglip}

    def sample_transforms(self, batch_size: int, patch_size: int) -> list[SampleTransform]:
        specs = []
        area = self.image_size * self.image_size
        for _ in range(batch_size):
            crop_scale = self._rand_uniform(self.generator, 0.9, 1.0)
            aspect = self._rand_uniform(self.generator, 0.95, 1.05)
            crop_h = min(self.image_size, max(1, round(math.sqrt(area * crop_scale / aspect))))
            crop_w = min(self.image_size, max(1, round(math.sqrt(area * crop_scale * aspect))))
            crop_top = int(torch.randint(0, self.image_size - crop_h + 1, (), generator=self.generator).item())
            crop_left = int(torch.randint(0, self.image_size - crop_w + 1, (), generator=self.generator).item())
            local_scale = self._rand_uniform(self.generator, 0.8, 1.2)
            local_size = min(self.image_size, max(1, round(patch_size * local_scale)))
            patch_top = int(torch.randint(0, self.image_size - local_size + 1, (), generator=self.generator).item())
            patch_left = int(torch.randint(0, self.image_size - local_size + 1, (), generator=self.generator).item())
            specs.append(SampleTransform(
                crop_top, crop_left, crop_h, crop_w,
                self._rand_uniform(self.generator, 0.9, 1.1),
                self._rand_uniform(self.generator, 0.9, 1.1),
                local_size, local_size, patch_top, patch_left,
                self._rand_uniform(self.generator, 0.9, 1.1),
            ))
        return specs

    @staticmethod
    def apply_global_transform(image: torch.Tensor, spec: SampleTransform) -> torch.Tensor:
        transformed = TVF.resized_crop(
            image, spec.crop_top, spec.crop_left, spec.crop_height, spec.crop_width,
            [image.shape[-2], image.shape[-1]], antialias=True,
        )
        transformed = TVF.adjust_brightness(transformed, spec.brightness)
        return TVF.adjust_contrast(transformed, spec.contrast).clamp(0.0, 1.0)

    @staticmethod
    def apply_local_patch(image: torch.Tensor, patch: torch.Tensor, spec: SampleTransform) -> torch.Tensor:
        local_patch = F.interpolate(
            patch.unsqueeze(0), size=(spec.patch_height, spec.patch_width),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        local_patch = (local_patch * spec.patch_brightness).clamp(0.0, 1.0)
        mask = torch.zeros_like(image)
        padded = torch.zeros_like(image)
        top, left = spec.patch_top, spec.patch_left
        height, width = spec.patch_height, spec.patch_width
        mask[:, top:top + height, left:left + width] = 1.0
        padded[:, top:top + height, left:left + width] = local_patch
        return (1.0 - mask) * image + padded

    def paired_view(self, images, patch, specs):
        clean_views, adversarial_views = [], []
        for image, spec in zip(images, specs, strict=True):
            patched = self.apply_local_patch(image, patch, spec)
            clean_views.append(self.apply_global_transform(image, spec))
            adversarial_views.append(self.apply_global_transform(patched, spec))
        return torch.stack(clean_views), torch.stack(adversarial_views)

    def apply_baseline_patch(self, images: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        patch_size = patch.shape[-1]
        outputs = []
        for image in images:
            top = int(torch.randint(0, image.shape[-2] - patch_size + 1, (), generator=self.generator).item())
            left = int(torch.randint(0, image.shape[-1] - patch_size + 1, (), generator=self.generator).item())
            mask, padded = torch.zeros_like(image), torch.zeros_like(image)
            mask[:, top:top + patch_size, left:left + patch_size] = 1.0
            padded[:, top:top + patch_size, left:left + patch_size] = patch
            outputs.append((1.0 - mask) * image + padded)
        return torch.stack(outputs)

    def compute_edpa_loss(self, adversarial, clean, text, text_mask):
        logits = torch.bmm(
            F.normalize(adversarial, dim=-1), F.normalize(clean, dim=-1).transpose(1, 2)
        ) / 0.07
        labels = torch.arange(logits.size(1), device=logits.device).unsqueeze(0).expand(logits.size(0), -1)
        patch_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).reshape(logits.size(0), logits.size(1)).mean(dim=1)
        adversarial_similarity = torch.bmm(
            F.normalize(adversarial, dim=-1), F.normalize(text, dim=-1).transpose(1, 2)
        )
        clean_similarity = torch.bmm(
            F.normalize(clean, dim=-1), F.normalize(text, dim=-1).transpose(1, 2)
        )
        if text_mask is not None:
            mask = text_mask.float().unsqueeze(1).to(self.device_id)
            valid = torch.clamp(mask.sum(dim=-1), min=EPS)
            align_loss = torch.sum(torch.abs(adversarial_similarity - clean_similarity) * mask, dim=(1, 2))
            align_loss = align_loss / valid.squeeze(-1)
        else:
            align_loss = torch.mean(torch.abs(adversarial_similarity - clean_similarity), dim=(1, 2))
        decay = self.ema_decay * min(self.ema_step / self.ema_warmup, 1.0)
        ema_patch = (decay * self.ema_patch + (1.0 - decay) * patch_loss.mean()).detach()
        ema_align = (decay * self.ema_align + (1.0 - decay) * align_loss.mean()).detach()
        normalized_patch = patch_loss / (ema_patch + EPS)
        normalized_align = align_loss / (ema_align + EPS)
        objective = self.alpha * normalized_patch + (1.0 - self.alpha) * normalized_align
        return objective.mean(), {
            "patch_raw": patch_loss.mean().detach(), "align_raw": align_loss.mean().detach(),
            "patch_normalized": normalized_patch.mean().detach(),
            "align_normalized": normalized_align.mean().detach(),
            "ema_patch": ema_patch, "ema_align": ema_align,
        }

    @staticmethod
    def gradient_norm(gradient: torch.Tensor) -> float:
        return torch.linalg.vector_norm(gradient.float()).item()

    def optimize_step(self, vla, processor, images, instructions, text_mask, patch):
        image_processor = processor.image_processor
        with torch.no_grad():
            clean_features = self.get_img_embedding(vla, self.preprocess(images, image_processor)).detach()
            text_features = self.get_lang_embedding(vla, instructions).detach()

        patch_leaf = patch.detach().requires_grad_(True)
        baseline_images = self.apply_baseline_patch(images, patch_leaf)
        baseline_features = self.get_img_embedding(vla, self.preprocess(baseline_images, image_processor))
        edpa_objective, edpa_metrics = self.compute_edpa_loss(
            baseline_features, clean_features, text_features, text_mask
        )
        edpa_gradient = torch.autograd.grad(edpa_objective, patch_leaf)[0]

        view_specs = [self.sample_transforms(images.shape[0], patch.shape[-1]) for _ in range(self.num_views)]
        branches = ("dino", "siglip")
        clean_logits = {branch: [] for branch in branches}
        clean_probabilities = {branch: [] for branch in branches}
        references = {branch: [] for branch in branches}
        with torch.no_grad():
            reference_directions = {branch: [] for branch in branches}
            for specs in view_specs:
                clean_view, adversarial_view = self.paired_view(images, patch.detach(), specs)
                clean_by_branch = self.branch_features(vla, self.preprocess(clean_view, image_processor))
                adversarial_by_branch = self.branch_features(vla, self.preprocess(adversarial_view, image_processor))
                for branch in branches:
                    clean_relation, clean_probability, valid_mask = local_token_relations(
                        clean_by_branch[branch], self.relation_temperature
                    )
                    adversarial_relation, _, _ = local_token_relations(
                        adversarial_by_branch[branch], self.relation_temperature
                    )
                    clean_logits[branch].append(clean_relation.detach())
                    clean_probabilities[branch].append(clean_probability.detach())
                    reference_directions[branch].append(
                        relation_direction(adversarial_relation - clean_relation, valid_mask).detach()
                    )
            for branch in branches:
                references[branch] = F.normalize(
                    torch.stack(reference_directions[branch]).mean(dim=0), dim=-1, eps=EPS
                ).detach()

        component_gradients = []
        trd_values = {branch: [] for branch in branches}
        consistency_values = {branch: [] for branch in branches}
        branch_consistency_values = []
        for view_index, specs in enumerate(view_specs):
            view_patch = patch.detach().requires_grad_(True)
            _, adversarial_view = self.paired_view(images, view_patch, specs)
            adversarial_by_branch = self.branch_features(vla, self.preprocess(adversarial_view, image_processor))
            direction_by_branch = {}
            component_losses = []
            for branch in branches:
                adversarial_relation, adversarial_probability, valid_mask = local_token_relations(
                    adversarial_by_branch[branch], self.relation_temperature
                )
                direction = relation_direction(
                    adversarial_relation - clean_logits[branch][view_index], valid_mask
                )
                direction_by_branch[branch] = direction
                trd = jensen_shannon_map(
                    clean_probabilities[branch][view_index], adversarial_probability
                ).mean()
                consistency = F.cosine_similarity(direction, references[branch], dim=-1).mean()
                trd_values[branch].append(trd.detach())
                consistency_values[branch].append(consistency.detach())
                component_losses.extend([trd, consistency])
            branch_consistency = F.cosine_similarity(
                direction_by_branch["dino"], direction_by_branch["siglip"], dim=-1
            ).mean()
            branch_consistency_values.append(branch_consistency.detach())
            component_losses.append(branch_consistency)
            for loss_index, loss in enumerate(component_losses):
                component_gradients.append(torch.autograd.grad(
                    loss, view_patch, retain_graph=loss_index < len(component_losses) - 1
                )[0])

        auxiliary_gradient, reliability_weights, gradient_cosine_mean, gradient_cosine_min = (
            reliability_weighted_gradient(component_gradients)
        )
        projected_auxiliary, anchor_dot_before, anchor_dot_after, projection_applied = (
            scale_and_project_to_anchor(auxiliary_gradient, edpa_gradient)
        )
        total_gradient = edpa_gradient + projected_auxiliary
        update_direction = total_gradient.sign()
        update_anchor_dot = torch.sum(update_direction.float() * edpa_gradient.float()).item()
        fallback_to_edpa = update_anchor_dot < 0.0
        if fallback_to_edpa:
            update_direction = edpa_gradient.sign()
            update_anchor_dot = torch.sum(update_direction.float() * edpa_gradient.float()).item()
        updated_patch = torch.clamp(patch + self.step_size * update_direction, 0.0, 1.0).detach()

        self.ema_patch = edpa_metrics["ema_patch"]
        self.ema_align = edpa_metrics["ema_align"]
        self.ema_step += 1
        metrics = {
            "objective_edpa": edpa_objective.detach().item(),
            **{name: value.item() for name, value in edpa_metrics.items()},
            "grad_norm_edpa": self.gradient_norm(edpa_gradient),
            "grad_norm_auxiliary_unit_aggregate": self.gradient_norm(auxiliary_gradient),
            "grad_norm_auxiliary_projected": self.gradient_norm(projected_auxiliary),
            "grad_norm_total": self.gradient_norm(total_gradient),
            "component_gradient_cosine_mean": gradient_cosine_mean,
            "component_gradient_cosine_min": gradient_cosine_min,
            "reliability_weight_min": min(reliability_weights),
            "reliability_weight_max": max(reliability_weights),
            "reliability_active_components": sum(weight > 0.0 for weight in reliability_weights),
            "auxiliary_edpa_dot_before": anchor_dot_before,
            "auxiliary_edpa_dot_after": anchor_dot_after,
            "anchor_projection_applied": projection_applied,
            "update_edpa_dot": update_anchor_dot,
            "fallback_to_edpa_sign": fallback_to_edpa,
            "branch_relation_consistency_raw": torch.stack(branch_consistency_values).mean().item(),
        }
        for branch in branches:
            metrics[f"{branch}_token_relation_js_raw"] = torch.stack(trd_values[branch]).mean().item()
            metrics[f"{branch}_view_relation_consistency_raw"] = torch.stack(
                consistency_values[branch]
            ).mean().item()
        metrics["token_relation_js_raw"] = 0.5 * (
            metrics["dino_token_relation_js_raw"] + metrics["siglip_token_relation_js_raw"]
        )
        return updated_patch, metrics

    def save(self, patch: torch.Tensor) -> None:
        torch.save(patch, self.output_dir / "perturbation.pt")
        save_image(patch, self.output_dir / "perturbation.png")

    def generate(self, vla, dataloader, processor, action_tokenizer=None):
        del action_tokenizer
        patch_size = int(math.sqrt(self.image_size**2 * self.perturbation_ratio))
        patch = torch.rand((3, patch_size, patch_size), generator=self.generator)
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            with tqdm(total=self.max_steps, leave=False) as progress:
                for step, batch in enumerate(dataloader):
                    if step >= self.max_steps:
                        break
                    images = self.convert_to_tensor(batch["images"])
                    patch, metrics = self.optimize_step(
                        vla, processor, images, batch["input_ids"], batch["attention_mask"], patch
                    )
                    metrics_file.write(json.dumps({"step": step, **metrics}, ensure_ascii=True) + "\n")
                    metrics_file.flush()
                    if step % self.save_steps == 0:
                        self.save(patch)
                    progress.set_postfix(
                        edpa=f"{metrics['objective_edpa']:.3f}",
                        trd=f"{metrics['token_relation_js_raw']:.3f}",
                        bc=f"{metrics['branch_relation_consistency_raw']:.3f}",
                    )
                    progress.update()
                    torch.cuda.empty_cache()
        self.save(patch)
        return patch.cpu()
