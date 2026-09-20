#!/usr/bin/env python3
"""Generate a full single-SigLIP TC-TD V3 Pi0 patch for one camera."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import jax.numpy as jnp
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.shared import download
from openpi.training import config

from tc_td_v3_pi0 import Pi0TokenRelationV3
from utils.data_utils import RLDSDataLoader


@dataclass
class Config:
    model_family: str = "pi0"
    vla_path: str = os.environ.get("PI0_CHECKPOINT_PATH", "checkpoints/pi0_libero")
    policy_config_name: str = os.environ.get("PI0_POLICY_CONFIG_NAME", "pi0_libero")
    data_root_dir: Path = Path("dataset/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    image_size: int = 224
    perturbation_ratio: float = 0.05
    alpha: float = 0.8
    max_steps: int = 50000
    iterations: int = 1
    step_size: float = 2 / 255
    save_path: str = ""
    batch_size: int = 16
    save_steps: int = 10
    shuffle_buffer_size: int = 2000
    camera_view: str = "primary"
    cycle_dataloader: bool = True
    seed: int = 7
    use_wandb: bool = False
    run_id_note: Optional[str] = None
    num_views: int = 4
    relation_temperature: float = 0.1


@draccus.wrap()
def main(cfg: Config) -> None:
    if cfg.batch_size < 1 or cfg.num_views < 2 or cfg.relation_temperature <= 0:
        raise ValueError("Invalid V3 batch/view/temperature configuration")
    dataloader = RLDSDataLoader(cfg=cfg)
    checkpoint = download.maybe_download(cfg.vla_path)
    train_config = config.get_config(cfg.policy_config_name)
    model = train_config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=train_config.model.max_token_len)
    Pi0TokenRelationV3(cfg).generate(model, dataloader, tokenizer)


if __name__ == "__main__":
    main()
