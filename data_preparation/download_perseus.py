"""
Inspired by

https://github.com/brennannicholson/ancient-greek-char-bert/blob/master/data_prep/greek_data_prep/clean_data.py
"""

import os
import re
import random
import shutil

from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

from data_preparation.data_prep_utils import (
    get_files,
    clean_texts,
    download_git_repo,
    split_texts,
    print_stats_and_save
)
from data_preparation.download_all import MLM_TARGET_DIR

# e