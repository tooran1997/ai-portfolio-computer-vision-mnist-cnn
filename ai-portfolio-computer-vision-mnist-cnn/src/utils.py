"""
Utility helpers for training & evaluation.
"""
from typing import Tuple
import torch
from torch.utils.data import random_split

def train_val_split(dataset, val_ratio: float = 0.1, seed: int = 42) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)
