
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
        Hugging Face and returns them."""
    train_losses, val_losses = [], []

    # scan tensorboard files and save the losses in the lists
    tb_files = glob.glob(f'{logdir}/events.out.tfevents.*') + \
        glob.glob(f'{logdir}/*/events.out.tfevents.*')
    for tb_out in tb_files:
        for e in EventFileLoader(tb_out).Load():
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/loss':
                    train_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'eval/loss':
                    val_losses.append(e.summary.value[0].tensor.float_val[0])

    return train_losses, val_losses


def plot_mlm_losses(
        logdir: Path,
        savepath: Path,
        framework: str = 'pl',
        test_loss: Optional[float] = None
) -> None:
    """Plots the train loss per batch step, the validation loss per epoch
        (which is equivalent to the batch steps that correspond to one epoch)
        and the test loss of the final model (basically scattered as a point)
        and saves the result in the specified file."""
    if framework == 'pl':
        train_losses, val_losses, test_loss = get_pl_mlm_losses(logdir)
    elif framework == 'hf':
        train_losses, val_losses = get_hf_mlm_losses(logdir)
    else:
        raise ValueError(f'Unknown framework {framework}.')

    # determine the batch step index of the epochs ends
    val_steps = np.linspace(0, len(train_losses) - 1, len(val_losses) + 1)[1:]

    # plot
    plt.figure(figsize=(16, 12))
    plt.plot(train_losses, label='Batch Train Loss', linewidth=2, color='blue')
    plt.plot(val_steps, val_losses, label='Epoch Validation Loss', linewidth=2,
             color='orange')
    if test_loss is not None:
        plt.scatter(len(train_losses) - 1, test_loss, label='Test Loss',
                    color='red', s=45)
    plt.xlabel('Steps', fontsize=16)
    plt.xticks(fontsize='12')
    plt.ylabel('Loss', fontsize=16)
    plt.yticks(fontsize='12')
    plt.title('Learning curves', fontsize=16)
    plt.legend(prop={'size': 16})

    # save
    if not savepath.parent.is_dir():
        os.makedirs(savepath.parent)
    plt.savefig(savepath, bbox_inches='tight')


def get_pl_pos_metrics(logdir: Path) -> Tuple[
    Tuple[List[float], List[float], List[float]],
    Tuple[List[float], List[float], List[float]],
    Tuple[Optional[float], Optional[float], Optional[float]]
]:
    """Reads the train/val/test metrics (loss/accuracy/f1) from tensorboard
        log files and returns them."""
    train_losses, val_losses, test_loss = [], [], None
    train_acc, val_acc, test_acc = [], [], None
    train_f1, val_f1, test_f1 = [], [], None

    # scan tensorboard files and save the losses in the lists
    tb_files = glob.glob(f'{logdir}/*/*/events.out.tfevents.*')
    for tb_out in tb_files:
        for e in EventFileLoader(tb_out).Load():
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/batch_loss':
                    train_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'train/batch_acc':
                    train_acc.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'train/batch_f1':
                    train_f1.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'val/val_loss':
                    val_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'val/val_acc':
                    val_acc.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'val/val_f1':
                    val_f1.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'test/test_loss':
                    test_loss = e.summary.value[0].tensor.float_val[0]
                elif e.summary.value[0].tag == 'test/test_acc':
                    test_acc = e.summary.value[0].tensor.float_val[0]
                elif e.summary.value[0].tag == 'test/test_f1':
                    test_f1 = e.summary.value[0].tensor.float_val[0]

    train_metrics = (train_losses, train_acc, train_f1)
    val_metrics = (val_losses, val_acc, val_f1)
    test_metrics = (test_loss, test_acc, test_f1)
    return train_metrics, val_metrics, test_metrics


def get_hf_pos_metrics(logdir: Path) -> Tuple[
    Tuple[List[float], List[float], List[float]],
    Tuple[List[float], List[float], List[float]]
]:
    """Reads the train/val loss/accuracy/f1 metrics from tensorboard files
        and returns them."""
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    train_f1, val_f1 = [], []

    # scan tensorboard files and save the losses in the lists
    tb_files = glob.glob(f'{logdir}/events.out.tfevents.*') + \
        glob.glob(f'{logdir}/*/events.out.tfevents.*')
    for tb_out in tb_files:
        for e in EventFileLoader(tb_out).Load():
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/loss':
                    train_losses.append(e.summary.value[0].tensor.float_val[0])
                elif e.summary.value[0].tag == 'eval/loss':
                    val_losses.append(e.summary.value[0].tensor.float_val[0])