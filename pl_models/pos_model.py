
import torch
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path
from torch.optim import AdamW
from typing import Tuple, List, Dict, Union, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau