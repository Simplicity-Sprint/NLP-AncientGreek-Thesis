import ast
import torch
import random
import configparser

from pathlib import Path
from typing import Dict, Union


def device_from_str(device_str: str) -> str:
    """Fixes the torch device string if needed."""
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    r