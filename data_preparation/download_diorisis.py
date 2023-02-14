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
                              desc='Converting POS data to MLM format'):
        doc_sentences = []
        for sentence, tags in zip(doc, doc_tags):
            sentence_as_str = sentence[0]
            for token, tag in zip(sentence[1:], tags[1:]):
                if tag != 'punct':
                    sentence_as_str += ' '
                sentence_as_str += token
            doc_sentences.append(sentence_as_str)
        mlm_sentences.append(doc_sentences)
    return mlm_sentences


def download_diorisis(mlm_dest_dir: Path, pos_dest_dir: Path) -> None:
    train_sentences, val_sentences, test_sentences = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    random.seed(80085)

    url = 'https://figshare.com/ndownloader/files/27831960'
    temp_dest = Path('temp-diorisis')
    download_and_unzip(url, temp_dest)
    if os.path.isfile(temp_dest/'corpus.json'):
        os.unlink(temp_dest/'corpus.json')

    num_invalid_sentences = 0

    # get all the JSON files with data and loop through them
    json_files = sorted(list(temp_dest.glob('*.json')))
    for json_file in tqdm(json_files, desc='Processing Diorisis Data'):

        # read the contents of the JSON file
        with open(json_file, 'r') as fp:
            contents = json.load(fp)

        # for each sentence, get the sub-words and tags and add them to the data
        doc_sentences, doc_pos_tags = [], []
        for sentence in contents['sentences']:
            sentence_text, sentence_pos_tags = get_tokens_and_pos_tags(sentence)
            if len(sentence_text) == 0:
                num_invalid_sentences += 1
                continue

            # for some reason, diorisis ends sentences at semicolons,
            #  but we don't want that, so concatenate the current sentence
            #  to the previous if the previous ends with a semicolon (·)
            if len(doc_sentences) > 0 and doc_sentences[-1][-1] == '·':
                doc_sentences[-1] += sentence_text
                doc_pos_tags[-1] += sentence_pos_tags
            else:
                doc_sentences.append(sentence_text)
                doc_pos_tags.append(sentence_pos_tags)

        # decide whether this document will be included (train/val/test)
        prob = random.uniform(0, 1)

        # 95 - 2.5 - 2.5 split (almost, because different documents
        # contain different number of sentences)
        if prob < 0.95:
            train_sentences.append(doc_sentences)
            train_labels.append(doc_pos_tags)
        elif prob < 0.975:
            val_sentences.append(doc_sentences)
            val_labels.append(doc_pos_tags)
        else:
            test_sentences.append(doc_sentences)
            test_labels.append(doc_pos_tags)

    print()
    print(f'Number of invalid sentences: {num_invalid_sentences}')
    print()

    # preprocess them a bit in order to be used also as MLM data
    train_mlm_sentences = convert_to_mlm_format(train_sentences, train_labels)
  