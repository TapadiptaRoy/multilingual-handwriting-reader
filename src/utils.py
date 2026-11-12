"""
utils.py
Small helper functions used across the project.
"""

import yaml


def load_config(path="configs/config.yaml"):
    """Read the YAML settings file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)