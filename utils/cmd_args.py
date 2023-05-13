
import argparse

from pathlib import Path
from typing import List, Optional


def parse_pl_mlm_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the PyTorch Lightning MLM training script."""
    default_logdir = Path(__file__).parent.parent/'logs'/'pl-mlm'
    default_savedir = Path(__file__).parent.parent/'objects'/'PL-AG-RoBERTa'
    default_seed = 'random'

    parser = argparse.ArgumentParser(description='PL MLM Training script.')

    # logs directory
    parser.add_argument('-l', '--logdir', type=Path, action='store',
                        metavar='logs-directory', default=default_logdir,
                        help='Path to the tensorboard logs directory.')
    # optional config file
    parser.add_argument('-c', '--config-path', type=Path, action='store',
                        metavar='configuration-file',
                        help='Path to the configuration file that will be '
                             'used to set the hyperparameters of the model.')
    # path to save the directory where the model will be saved
    parser.add_argument('-s', '--savedir', type=Path, action='store',
                        default=default_savedir, metavar='model-save-directory',
                        help='Path to the directory where the pre-trained '
                             'model will be saved.')
    # optional path to save plot with learning curves
    parser.add_argument('-p', '--plot-savepath', type=Path, action='store',
                        metavar='path-to-save-learning-curves-plot',
                        help='Path to the a .png filename where the learning '