import torch
import argparse
import pytorch_lightning as pl

from transformers import RobertaTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.pos_model import PoSRoBERTa
from utils.cmd_args import parse_pl_pos_input
from utils.plot_utils import plot_pos_metrics
from ag_datasets.pos_dataset import PoSDataset
from utils.fs_utils import force_empty_directory
from utils.run_utils import device_from_str, get_seed, hyperparams_from_config
from data_preparation.processing import (
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
    PROCESSED_DATA_PATH
)


def main(args: argparse.Namespace):
    """main() driver function."""

    # fix some args
    device_str = device_from_str(args.device)
    if args.distributed is True and device_str == 'cpu':
        raise RuntimeError("Distributed training can needs CUDA.")
    gpus = torch.cuda.device_count() if args.distributed is True else \
        1 if device_str == 'cuda' else None
    distributed_strategy = 'ddp' if args.distributed is True else None
    seed = get_seed(args.seed)

    # empty the tensorboard and model directories
    force_empty_directory(args.logdir)
    force_empty_directory(args.savedir)

    # load the tokenizer and fix the random seed
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    pl.seed_everything(seed)

    # define the default hyperparameters for the model here
    custom_hyperparameters = {
        'max-length': 512,
        'batch-size': 4,
        'learning-rate': 1e-4,
        'weight-decay': 1e-2,
        'use-lr-scheduler': True,
        'scheduler-factor': 0.1,
        'scheduler-patience': 10,
        'scheduler-step-update': 10,
