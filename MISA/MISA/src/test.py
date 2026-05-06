import os
import torch
import torch.nn as nn
import numpy as np
from random import random
from sklearn.metrics import f1_score, classification_report, accuracy_score

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver
from utils import to_gpu
from lora import merge_lora_weights, get_lora_params
import models

import warnings
warnings.filterwarnings('ignore')


def get_checkpoint_path(config):
    """Return the default checkpoint path based on dataset."""
    if config.data == 'mosi':
        return 'checkpoints/best_model_mosi.std'
    elif config.data == 'mosei':
        return 'checkpoints/best_model_mosei.std'
    elif config.data == 'ur_funny':
        return 'checkpoints/best_model_ur_funny.std'
    else:
        raise ValueError(f"Unknown dataset: {config.data}")


def load_checkpoint(model, path, config):
    """
    Load checkpoint into model, handling both merged and unmerged LoRA checkpoints.
    If the checkpoint was saved after LoRA merging (merge_lora_at_end=True during
    training), the state dict contains plain weight keys and loads directly.
    Otherwise, it contains lora_A / lora_B keys that map to LoRALinear layers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    state_dict = torch.load(path, map_location='cpu')

    # Detect whether the checkpoint has LoRA keys
    has_lora_keys = any('lora_' in k for k in state_dict.keys())

    if config.use_lora and not has_lora_keys:
        # Checkpoint was saved after weight merging — load directly
        print("[LoRA] Checkpoint contains merged weights. Loading directly.")
        model.load_state_dict(state_dict, strict=False)
    elif config.use_lora and has_lora_keys:
        # Checkpoint retains separate LoRA matrices — load normally
        print("[LoRA] Checkpoint contains unmerged LoRA weights. Loading directly.")
        model.load_state_dict(state_dict)
    else:
        # Non-LoRA checkpoint
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':

    # Setting random seed for reproducibility
    random_seed = 337
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # ── Configuration ──────────────────────────────────────────────────────────
    train_config = get_config(mode='train')
    dev_config   = get_config(mode='dev')
    test_config  = get_config(mode='test')

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(test_config)

    if test_config.use_lora:
        print("\n" + "=" * 70)
        print("LoRA CONFIGURATION")
        print("=" * 70)
        print(f"  LoRA Rank        : {test_config.lora_rank}")
        print(f"  LoRA Alpha       : {test_config.lora_alpha}")
        print(f"  LoRA Dropout     : {test_config.lora_dropout}")
        print(f"  Train Only LoRA  : {test_config.train_only_lora}")
        print(f"  Merge LoRA at End: {test_config.merge_lora_at_end}")
        print("=" * 70 + "\n")

    # ── Data loaders ───────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_data_loader = get_loader(train_config, shuffle=True)
    dev_data_loader   = get_loader(dev_config,   shuffle=False)
    test_data_loader  = get_loader(test_config,  shuffle=False)
    print(f"Train batches : {len(train_data_loader)}")
    print(f"Dev batches   : {len(dev_data_loader)}")
    print(f"Test batches  : {len(test_data_loader)}\n")

    # ── Model instantiation ────────────────────────────────────────────────────
    model = getattr(models, test_config.model)(test_config)

    if test_config.use_lora:
        lora_params  = get_lora_params(model)
        total_params = sum(p.numel() for p in model.parameters())
        lora_count   = sum(p.numel() for p in lora_params)
        print(f"[LoRA] Total parameters : {total_params:,}")
        print(f"[LoRA] LoRA parameters  : {lora_count:,} "
              f"({100 * lora_count / total_params:.2f}%)\n")

    # ── Checkpoint loading ─────────────────────────────────────────────────────
    checkpoint_path = get_checkpoint_path(test_config)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, test_config)

    # ── Optional: merge LoRA weights into base weights before inference ────────
    # Merging eliminates the LoRA overhead at inference time (no accuracy change).
    if test_config.use_lora and test_config.merge_lora_at_end:
        print("[LoRA] Merging LoRA weights into base weights for inference...")
        merge_lora_weights(model)
        print("[LoRA] Merge complete.\n")

    model.eval()

    # ── Solver & evaluation ────────────────────────────────────────────────────
    solver = Solver(
        train_config, dev_config, test_config,
        train_data_loader, dev_data_loader, test_data_loader,
        is_train=False,
        model=model
    )

    solver.build()
    solver.criterion = nn.MSELoss(reduction="mean")

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    solver.eval(mode="test", to_print=True)
    print("=" * 70 + "\n")