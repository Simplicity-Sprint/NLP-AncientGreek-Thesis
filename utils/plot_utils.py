
import os
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple, Optional
from tensorboard.backend.event_processing.event_file_loader import \
    EventFileLoader


def get_pl_mlm_losses(logdir: Path) -> \
        Tuple[List[float], List[float], Optional[float]]:
    """Reads the train/val/test losses from tensorboard log files produced by
        PyTorch Lightning and returns them."""
    train_losses, val_losses, test_loss = [], [], None

    # scan tensorboard files and save the losses in the lists
    tb_files = glob.glob(f'{logdir}/*/*/events.out.tfevents.*')
    for tb_out in tb_files:
        for e in EventFileLoader(tb_out).Load():
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/batch_loss':
                    train_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'val/val_loss':
                    val_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'test/test_loss':
                    test_loss = e.summary.value[0].tensor.float_val[0]

    return train_losses, val_losses, test_loss


def get_hf_mlm_losses(logdir: Path) -> Tuple[List[float], List[float]]:
    """Reads the train/val losses from tensorboard log files produced by