import os
import json
import shutil
import random
import betacode.conv

from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Union


from data_preparation.data_prep_utils import (
    clean_raw_text,
    download_and_unzip,
    print_stats_and_save,
    save_pickle
)
from data_preparation.download_all import MLM_TARGET_DIR, POS_TARGET_DIR


LEGIT_PUNCT = ['.’', '/', '—', ',’', '’', '.', ';', '%', '(', ')', ',', '‘',
               '·', '«', '»', '"']


def get_tokens_and_pos_tags(
        sentence_dict: Dict[
            str,
            Union[str, List[Dict[
                str,
                Union[str, Union[str, Dict[str, List[str]]]]]]
            ]
        ]
) -> Tuple[List[str], List[str]]:
    """Given a sentence dictionary, it returns the text parts inside