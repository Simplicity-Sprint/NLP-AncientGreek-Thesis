import time
import torch
import logging
import argparse
import warnings
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from functools import partial
from typing import Tuple, Dict, Union
from transformers import RobertaTokenizerFast
from pytorch_lightning import seed_everything
from hyperopt import hp, fmin, tpe, space_eval
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.mlm_model import LitRoBERTaMLM
from utils.plot_utils import get_pl_mlm_losses
from utils.cmd_args import parse_tune_mlm_input
from utils.fs_utils import force_empty_directory, delete_file_if_exists
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH


BEST_VAL_LOSS = float('inf')
BEST_ARGS = None


def create_and_train_model(
        args: Dict[str, Union[float, int]],
        constants: Dict[str, Union[int, float, bool, Tuple[Path, Path, Path],
                                   RobertaTokenizerFast, Path]]
) -> LitRoBERTaMLM:
    """Creates and pre-trains a PL MLM Ancient Greek RoBERTa model."""
    # set the seed
    seed_everything(args['seed'])

    # create PL model
    model = LitRoBERTaMLM(
        tokenizer=constants['tokenizer'],
        paths=constants['data-paths'],
        hyperparams={**args, **constants}
    )

    # handle logging
    logdir = constants['tb-logdir']
    force_empty_directory(logdir)
    logger = TensorBoardLogger(str(logdir), name='AG-RoBERTa-Temp', version=0)

    # train the model (use a good GPU, otherwise it will take ages, trust me)
    trainer = pl.Trainer(
        default_root_dir=str(logdir),
        gpus=1 if torch.cuda.is_available() else 0,
        # gpus=torch.cuda.device_count(),  # doesn't work for many, strange bug
        # strategy='ddp',
        max_epochs=constants['train-epochs'],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    trainer.fit(model)

    return model


def objective(
        args: Dict[str, Union[float, int]],
        constants: Dict[str, Union[int, float, bool, Tuple[Path, Path, Path],
                        RobertaTokenizerFast, Path]]
) -> float:
    """Creates, trains a PL MLM Ancient Greek RoBERTa model and returns its
        best loss (across all epochs) on the validation set."""
    # the hidden size must be a multiple of the number of attention heads
    hidden_size = args['hidden-size']
    num_attention_heads = args['num-attention-heads']
    hidden_size = (hidden_size // num_attention_heads) * num_attention_heads
    args['hidden-size'] = hidden_size

    # train the model (which is automatically evaluated at every epoch)
    create_and_train_model(args, constants)

    # get the validation losses for every epoch from the tensorboard logs
    _, val_losses, _ = get_pl_mlm_losses(constants['tb-logdir'])

    # the performance of the model is the best (minimum) validation loss
    performance = min(val_losses)

    # compare value to global best
    global BEST_VAL_LOSS, BEST_ARGS
    if performance < BEST_VAL_LOSS:
        BEST_VAL_LOSS = performance
        BEST_ARGS = args

    # write it on the output file so that we can see the results real time
    with open(constants['tune-logfile'], 'a') as fp:
        fp.write(f'For hyperparameters: {args}\n'
                 f'The validation loss is {performance}.\n\n'
                 f'The best hyperparameters so far are {BEST_ARGS}\n'
                 f'Which give a validation loss of {BEST_VAL_LOSS}.\n\n\n\n')

    return performance


def main(args: argparse.Namespace):

    # define the constant values of the model
    data_dir = PROCESSED_DATA_PATH/'MLM'
    data_paths = (data_dir/'train-data.pkl',
                  data_dir/'val-data.pkl',
                  data_dir/'test-data.pkl')
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    tb_logdir = Path('logs')/'pl-mlm-hp-tuning'
    tune_logfile = Path('logs')/'pl-mlm-hp-tuning-results.txt'
    delete_file_if_exists(tune_logfile)
    constants = {
        'max-length': 512,
        'mask-probability': 0.15,
        'type-vocab-size': 1,
        'use-lr-scheduler': True,
        'scheduler-fa