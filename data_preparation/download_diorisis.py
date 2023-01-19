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
    """Given a sentence dictionary, it returns the text parts inside a list,
        and the corresponding labels for each string in another list."""
    # lists to hold individual tokens and POS tags
    tokens, labels = [], []
    # iterate through all the tokens
    for token in sentence_dict['tokens']:
        # invalid entries are not punct and don't have a lemma ->
        #  whole sentence is broken, skip it
        if token['type'] != 'punct' and \
                ('lemma' not in token or 'POS' not in token['lemma']):
            return [], []
        elif 'lemma' in token and token['lemma'].get('POS', None) == '':
            return [], []
        elif token['form'] == '':
            return [], []
        elif token['type'] == 'pu