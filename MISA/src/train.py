import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings
import models
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 337
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(train_config)
    if train_config.use_lora:
        print("\n" + "="*70)
        print("LoRA CONFIGURATION")
        print("="*70)
        print(f"LoRA Rank: {train_config.lora_rank}")
        print(f"LoRA Alpha: {train_config.lora_alpha}")
        print(f"LoRA Dropout: {train_config.lora_dropout}")
        print(f"Train Only LoRA: {train_config.train_only_lora}")
        print(f"Merge LoRA at End: {train_config.merge_lora_at_end}")
        print("="*70 + "\n")

    # Creating pytorch dataloaders
    print("Loading datasets...")
    train_data_loader = get_loader(train_config, shuffle=True)
    dev_data_loader = get_loader(dev_config, shuffle=False)
    test_data_loader = get_loader(test_config, shuffle=False)
    print(f"Train batches: {len(train_data_loader)}")
    print(f"Dev batches: {len(dev_data_loader)}")
    print(f"Test batches: {len(test_data_loader)}\n")

    # Solver is a wrapper for model training and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70 + "\n")
