"""Single-SigLIP TC-TD V3 optimizer for the JAX Pi0 policy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from tqdm import tqdm

from utils.pi0 import get_img_embedding, get_lang_embedding, image_transform


EPS = 1e-8
OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1), (-2, 0), (0, -2), (0, 2), (2, 0),
)


@dataclass(frozen=True)
class TransformSpec:
    crop_top: int
    crop_left: int
    crop_height: int
    crop_width: int
    brightness: float
    contrast: float
    patch_size: int
    patch_top: int
    patch_left: int
    patch_brightness: float


def normalize(value: jax.Array, axis: int = -1) -> jax.Array:
    return value / (jnp.linalg.norm(value, axis=axis, keepdims=True) + EPS)


def local_relations(features: jax.Array, temperature: float):
    if features.ndim != 3:
        raise ValueError(f"Expected [batch,tokens,dim], got {features.shape}")
    batch, tokens, dim = features.shape
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError(f"Pi0 token count {tokens} is not a square grid")
    values = normalize(features.astype(jnp.float32)).reshape(batch, side, side, dim)
    rows = jnp.arange(side)[:, None]
    cols = jnp.arange(side)[None, :]
    similarities, masks = [], []
    for dy, dx in OFFSETS:
        shifted = jnp.roll(values, shift=(-dy, -dx), axis=(1, 2))
        similarities.append(jnp.sum(values * shifted, axis=-1))
        masks.append((rows + dy >= 0) & (rows + dy < side) & (cols + dx >= 0) & (cols + dx < side))
    logits = jnp.stack(similarities, axis=-1).reshape(batch, tokens, -1)
    valid = jnp.stack(masks, axis=-1).reshape(1, tokens, -1)
    probabilities = jax.nn.softmax(jnp.where(valid, logits / temperature, -1e9), axis=-1)
    return jnp.where(valid, logits, 0.0), probabilities, valid


def js_divergence(p: jax.Array, q: jax.Array) -> jax.Array:
    p, q = jnp.maximum(p, EPS), jnp.maximum(q, EPS)
    midpoint = 0.5 * (p + q)
    return 0.5 * jnp.sum(
        p * (jnp.log(p) - jnp.log(midpoint)) + q * (jnp.log(q) - jnp.log(midpoint)), axis=-1
    )


def relation_direction(delta: jax.Array, valid: jax.Array) -> jax.Array:
    return normalize((delta * valid).reshape(delta.shape[0], -1))


class Pi0TokenRelationV3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.save_steps = cfg.save_steps
        self.step_size = cfg.step_size
        self.alpha = cfg.alpha
        self.image_size = cfg.image_size
        self.ratio = cfg.perturbation_ratio
        self.num_views = cfg.num_views
        self.temperature = cfg.relation_temperature
        self.ema_step = 0
        self.ema_patch = jnp.array(0.0)
        self.ema_align = jnp.array(0.0)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.output_dir = Path(f"{cfg.save_path}-{cfg.perturbation_ratio}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    @staticmethod
    def to_bchw(images) -> jax.Array:
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4:
            raise ValueError(f"Expected Pi0 images [B,H,W,C], got {images.shape}")
        return jnp.asarray(images).transpose(0, 3, 1, 2).astype(jnp.float32) / 255.0

    @staticmethod
    def preprocess(images: jax.Array) -> jax.Array:
        return jnp.stack([image_transform(jnp.transpose(image, (1, 2, 0))) for image in images])

    def sample_specs(self, batch: int, patch_size: int) -> list[TransformSpec]:
        result = []
        area = self.image_size * self.image_size
        for _ in range(batch):
            scale = self.rng.uniform(0.9, 1.0)
            aspect = self.rng.uniform(0.95, 1.05)
            crop_h = min(self.image_size, max(1, round(math.sqrt(area * scale / aspect))))
            crop_w = min(self.image_size, max(1, round(math.sqrt(area * scale * aspect))))
            local_size = min(self.image_size, max(1, round(patch_size * self.rng.uniform(0.8, 1.2))))
            result.append(TransformSpec(
                int(self.rng.integers(0, self.image_size - crop_h + 1)),
                int(self.rng.integers(0, self.image_size - crop_w + 1)), crop_h, crop_w,
                float(self.rng.uniform(0.9, 1.1)), float(self.rng.uniform(0.9, 1.1)),
                local_size,
                int(self.rng.integers(0, self.image_size - local_size + 1)),
                int(self.rng.integers(0, self.image_size - local_size + 1)),
                float(self.rng.uniform(0.9, 1.1)),
            ))
        return result

    def global_transform(self, image: jax.Array, spec: TransformSpec) -> jax.Array:
        crop = image[:, spec.crop_top:spec.crop_top + spec.crop_height, spec.crop_left:spec.crop_left + spec.crop_width]
        resized = jax.image.resize(crop, (3, self.image_size, self.image_size), method="linear")
        resized = resized * spec.brightness
        channel_mean = resized.mean(axis=(1, 2), keepdims=True)
        return jnp.clip((resized - channel_mean) * spec.contrast + channel_mean, 0.0, 1.0)

    def local_patch(self, image: jax.Array, patch: jax.Array, spec: TransformSpec) -> jax.Array:
        resized = jax.image.resize(patch, (3, spec.patch_size, spec.patch_size), method="linear")
        resized = jnp.clip(resized * spec.patch_brightness, 0.0, 1.0)
        return image.at[
            :, spec.patch_top:spec.patch_top + spec.patch_size, spec.patch_left:spec.patch_left + spec.patch_size
        ].set(resized)

    def paired_view(self, images: jax.Array, patch: jax.Array, specs: list[TransformSpec]):
        clean, adversarial = [], []
        for image, spec in zip(images, specs, strict=True):
            clean.append(self.global_transform(image, spec))
            adversarial.append(self.global_transform(self.local_patch(image, patch, spec), spec))
        return jnp.stack(clean), jnp.stack(adversarial)

    def baseline_patch(self, images: jax.Array, patch: jax.Array, positions: list[tuple[int, int]]):
        outputs = []
        size = patch.shape[-1]
        for image, (top, left) in zip(images, positions, strict=True):
            outputs.append(image.at[:, top:top + size, left:left + size].set(patch))
        return jnp.stack(outputs)

    def edpa_loss(self, adversarial, clean, text, text_mask):
        adv_norm, clean_norm, text_norm = normalize(adversarial), normalize(clean), normalize(text)
        logits = jnp.einsum("bij,bkj->bik", adv_norm, clean_norm) / 0.07
        labels = jnp.tile(jnp.arange(logits.shape[1])[None], (logits.shape[0], 1))
        patch_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        ).reshape(logits.shape[0], logits.shape[1]).mean(axis=1)
        adv_sim = jnp.einsum("bij,bkj->bik", adv_norm, text_norm)
        clean_sim = jnp.einsum("bij,bkj->bik", clean_norm, text_norm)
        mask = text_mask.astype(jnp.float32)[:, None, :]
        align_loss = jnp.sum(jnp.abs(adv_sim - clean_sim) * mask, axis=(1, 2)) / jnp.maximum(mask.sum(axis=-1).squeeze(-1), EPS)
        decay = 0.9 * min(self.ema_step / 100, 1.0)
        ema_patch = jax.lax.stop_gradient(decay * self.ema_patch + (1 - decay) * patch_loss.mean())
        ema_align = jax.lax.stop_gradient(decay * self.ema_align + (1 - decay) * align_loss.mean())
        objective = self.alpha * patch_loss / (ema_patch + EPS) + (1 - self.alpha) * align_loss / (ema_align + EPS)
        return objective.mean(), (ema_patch, ema_align, patch_loss.mean(), align_loss.mean())

    @staticmethod
    def aggregate_gradients(gradients: list[jax.Array]):
        flat = jnp.stack([normalize(gradient.reshape(-1), axis=0) for gradient in gradients])
        cosine = flat @ flat.T
        offdiag = ~jnp.eye(len(gradients), dtype=bool)
        reliability = jnp.where(offdiag, jnp.maximum(cosine, 0.0), 0.0).sum(axis=1) / max(len(gradients) - 1, 1)
        weights = jnp.where(reliability.sum() > EPS, reliability / reliability.sum(), jnp.ones_like(reliability) / len(gradients))
        return (flat * weights[:, None]).sum(axis=0).reshape(gradients[0].shape), weights, cosine

    def optimize_step(self, model, images, instructions, masks, patch):
        clean_embedding = jax.lax.stop_gradient(get_img_embedding(model, self.preprocess(images)))
        text_embedding = jax.lax.stop_gradient(get_lang_embedding(model, instructions))
        size = patch.shape[-1]
        positions = [
            (int(self.rng.integers(0, self.image_size - size + 1)), int(self.rng.integers(0, self.image_size - size + 1)))
            for _ in range(images.shape[0])
        ]

        def edpa_objective(value):
            adversarial = self.baseline_patch(images, value, positions)
            embedding = get_img_embedding(model, self.preprocess(adversarial))
            return self.edpa_loss(embedding, clean_embedding, text_embedding, masks)

        (edpa_value, edpa_aux), edpa_gradient = jax.value_and_grad(edpa_objective, has_aux=True)(patch)
        specs_by_view = [self.sample_specs(images.shape[0], size) for _ in range(self.num_views)]
        clean_relations, clean_probabilities, references = [], [], []
        for specs in specs_by_view:
            clean_view, adversarial_view = self.paired_view(images, patch, specs)
            clean_feature = get_img_embedding(model, self.preprocess(clean_view))
            adversarial_feature = get_img_embedding(model, self.preprocess(adversarial_view))
            clean_relation, clean_probability, valid = local_relations(clean_feature, self.temperature)
            adversarial_relation, _, _ = local_relations(adversarial_feature, self.temperature)
            clean_relations.append(jax.lax.stop_gradient(clean_relation))
            clean_probabilities.append(jax.lax.stop_gradient(clean_probability))
            references.append(jax.lax.stop_gradient(relation_direction(adversarial_relation - clean_relation, valid)))
        reference = jax.lax.stop_gradient(normalize(jnp.stack(references).mean(axis=0)))

        gradients, trd_values, consistency_values = [], [], []
        for index, specs in enumerate(specs_by_view):
            def components(value):
                _, adversarial_view = self.paired_view(images, value, specs)
                feature = get_img_embedding(model, self.preprocess(adversarial_view))
                relation, probability, valid = local_relations(feature, self.temperature)
                direction = relation_direction(relation - clean_relations[index], valid)
                trd = js_divergence(clean_probabilities[index], probability).mean()
                consistency = jnp.sum(direction * reference, axis=-1).mean()
                return jnp.stack([trd, consistency])
            values = components(patch)
            jacobian = jax.jacrev(components)(patch)
            gradients.extend([jacobian[0], jacobian[1]])
            trd_values.append(values[0]); consistency_values.append(values[1])

        auxiliary, weights, cosine = self.aggregate_gradients(gradients)
        anchor_norm = jnp.linalg.norm(edpa_gradient)
        auxiliary = auxiliary * anchor_norm / (jnp.linalg.norm(auxiliary) + EPS)
        dot_before = jnp.sum(auxiliary * edpa_gradient)
        auxiliary = jnp.where(
            dot_before < 0,
            auxiliary - dot_before / (jnp.sum(edpa_gradient ** 2) + EPS) * edpa_gradient,
            auxiliary,
        )
        total = edpa_gradient + auxiliary
        update = jnp.sign(total)
        update_dot = jnp.sum(update * edpa_gradient)
        update = jnp.where(update_dot < 0, jnp.sign(edpa_gradient), update)
        patch = jnp.clip(patch + self.step_size * update, 0.0, 1.0)
        self.ema_patch, self.ema_align = edpa_aux[0], edpa_aux[1]
        self.ema_step += 1
        offdiag = cosine[~jnp.eye(cosine.shape[0], dtype=bool)]
        metrics = {
            "objective_edpa": float(edpa_value),
            "patch_raw": float(edpa_aux[2]), "align_raw": float(edpa_aux[3]),
            "token_relation_js_raw": float(jnp.stack(trd_values).mean()),
            "view_relation_consistency_raw": float(jnp.stack(consistency_values).mean()),
            "branch_relation_consistency_raw": None,
            "grad_norm_edpa": float(anchor_norm),
            "grad_norm_auxiliary_projected": float(jnp.linalg.norm(auxiliary)),
            "component_gradient_cosine_mean": float(offdiag.mean()),
            "component_gradient_cosine_min": float(offdiag.min()),
            "reliability_weight_min": float(weights.min()),
            "reliability_weight_max": float(weights.max()),
            "auxiliary_edpa_dot_after": float(jnp.sum(auxiliary * edpa_gradient)),
            "update_edpa_dot": float(jnp.sum(update * edpa_gradient)),
        }
        return patch, metrics

    def save(self, patch: jax.Array):
        array = np.asarray(patch)
        np.save(self.output_dir / "perturbation.npy", array)
        image = np.clip(array.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(self.output_dir / "perturbation.png")

    def generate(self, model, dataloader, tokenizer):
        size = int(math.sqrt(self.image_size ** 2 * self.ratio))
        patch = jax.random.uniform(jax.random.PRNGKey(int(self.cfg.seed)), (3, size, size))
        step = 0
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            with tqdm(total=self.max_steps, leave=False) as progress:
                while step < self.max_steps:
                    made_progress = False
                    for batch in dataloader:
                        made_progress = True
                        if step >= self.max_steps:
                            break
                        key = "wrist_image" if self.cfg.camera_view == "wrist" else "image"
                        images = self.to_bchw(batch[key])
                        tokenized = [tokenizer.tokenize(value) for value in batch["language_instruction"]]
                        instructions = jnp.asarray([value[0] for value in tokenized])
                        masks = jnp.asarray([value[1] for value in tokenized])
                        patch, metrics = self.optimize_step(model, images, instructions, masks, patch)
                        metrics_file.write(json.dumps({"step": step, **metrics}, ensure_ascii=True) + "\n")
                        metrics_file.flush()
                        if step % self.save_steps == 0:
                            self.save(patch)
                        progress.set_postfix(
                            edpa=f"{metrics['objective_edpa']:.3f}", trd=f"{metrics['token_relation_js_raw']:.3f}"
                        )
                        progress.update(); step += 1
                    if not self.cfg.cycle_dataloader or not made_progress:
                        break
        if step != self.max_steps:
            raise RuntimeError(f"Pi0 dataloader ended at {step}/{self.max_steps}; enable cycle_dataloader")
        self.save(patch)
        return np.asarray(patch)
