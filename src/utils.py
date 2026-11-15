"""
utils.py
Small helper functions used across the project.
"""

import random

import numpy as np
import torch
import yaml


def load_config(path="configs/config.yaml"):
    """Read the YAML settings file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Fix all sources of randomness so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Use the GPU if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")