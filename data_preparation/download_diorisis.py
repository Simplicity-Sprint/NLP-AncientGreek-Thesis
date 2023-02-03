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
        elif token['type'] == 'punct' and token['form'] not in LEGIT_PUNCT:
            return [], []

        # get the actual Ancient Greek text, clean it, and get the pos tag
        token_text = clean_raw_text(betacode.conv.beta_to_uni(token['form']))
        pos = 'punct' if token['type'] == 'punct' else token['lemma']['POS']

        # add the entries to the lists
        tokens.append(token_text)
        labels.append(pos)

    return tokens, labels


def convert_to_mlm_format(
        sentences: List[List[List[str]]],
        pos_tags: List[List[List[str]]]
) -> List[List[str]]:
    """Converts the given sentences (which are in POS format) to MLM format,
        basically by concatenating each token with a space if the next token
        is not a punctuation character."""
    mlm_sentences = []
    for doc, doc_tags in tqdm(zip(sentences, pos_tags), total=len(sentences),
                              desc='Converting POS data to MLM format'