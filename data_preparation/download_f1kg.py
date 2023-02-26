
"""
Inspired by

https://github.com/brennannicholson/ancient-greek-char-bert/blob/master/data_prep/greek_data_prep/clean_data.py
"""

import os
import re
import glob
import shutil
import random

from pathlib import Path

from data_preparation.data_prep_utils import (
    download_and_unzip,
    clean_texts,
    get_files,
    split_texts,
    print_stats_and_save
)
from data_preparation.download_all import MLM_TARGET_DIR


def get_f1kg_texts(files):
    """Gets the specified F1KG text files (which do not need parsing,
        unlike the Perseus files)."""
    texts = []
    for i, f in enumerate(files):
        with open(f, 'r') as fp:
            texts.append(fp.read())
    return texts


def download_f1kg(dest_dir: Path) -> None: