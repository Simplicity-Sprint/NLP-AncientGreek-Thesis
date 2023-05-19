import ast
import torch
import random
import configparser

from pathlib import Path
from typing import Dict, Union


def device_from_str(device_str: str) -> str:
   