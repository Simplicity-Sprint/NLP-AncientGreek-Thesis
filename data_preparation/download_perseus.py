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

# exclude Bacchylides' Odes due to the fragmentary nature of the text
BACHCHYLIDES_ODES = [
    'tlg0199.tlg001.perseus-grc1.xml',
    'tlg0199.tlg002.perseus-grc1.xml',
]

TEXTS_WITH_SIGNIFICANT_OCR_ERRORS = [
    'tlg3129.ogl001.1st1K-grc1.xml',  # In Cyrilli In XII Prophetas Theophylacti
]

# ideally these should be automatically identified and converted
# but CLTK's parser doesn't seem to work...
BETA_CODE_FILES = [
    'tlg2003.tlg002.perseus-grc1.xml',
    'tlg2003.tlg010.perseus-grc1.xml',
    'tlg2003.tlg007.perseus-grc1.xml',
    'tlg2003.tlg009.perseus-grc1.xml',
    'tlg2003.tlg004.perseus-grc1.xml',
    'tlg2003.tlg012.perseus-grc1.xml',
    'tlg2003.tlg008.perseus-grc1.xml',
    'tlg2003.tlg005.perseus-grc1.xml',
    'tlg2003.tlg011.perseus-grc1.xml',
    'tlg2003.tlg001.perseus-grc1.xml',
    'tlg2003.tlg003