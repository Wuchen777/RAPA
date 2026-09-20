#!/usr/bin/env python3
"""Generate a full token-relation TC-TD V3 OpenVLA patch."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from tc_td_v3_attacker import TokenRelationTCTDV3Attacker
from utils.data_utils_openvla import PaddedCollatorForActionPrediction, RLDSBatchTransform, RLDSDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    vla_path: str = "openvla/openvla-7b-finetuned-libero-spatial"
    data_root_dir: Path = Path("dataset/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    image_size: int = 224
    perturbation_ratio: float = 0.05
    alpha: float = 0.8
    max_steps: int = 50000
    step_size: float = 2 / 255
    save_path: str = ""
    batch_size: int = 16
    save_steps: int = 10
    shuffle_buffer_size: int = 2000
    camera_view: str = "primary"
    seed: int = 7
    use_wandb: bool = False
    run_id_note: Optional[str] = None
    num_views: int = 4
    relation_temperature: float = 0.1
    embedding_backend: str = "openvla"


@draccus.wrap()
def main(cfg: Config) -> None:
    if cfg.batch_size < 1:
        raise ValueError("TC-TD V3 requires batch_size >= 1")
    if cfg.num_views < 2:
        raise ValueError("TC-TD V3 consistency requires num_views >= 2")
    if cfg.relation_temperature <= 0:
        raise ValueError("relation_temperature must be positive")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    state = PartialState()
    device_id = state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.seed)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    model.eval()
    model.requires_grad_(False)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        view=cfg.camera_view,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    attacker = TokenRelationTCTDV3Attacker(cfg, device_id=device_id)
    attacker.generate(model, dataloader, processor, action_tokenizer)


if __name__ == "__main__":
    main()
