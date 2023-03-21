
import os
import pickle
import re
import shutil

from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from sklearn.preprocessing import LabelEncoder

from data_preparation.download_all import MLM_TARGET_DIR, POS_TARGET_DIR


TOKENIZER_PATH = Path(__file__).parent.parent/'objects'/'bpe_tokenizer'
LABEL_ENCODER_PATH = Path(__file__).parent.parent/'objects'/'le.pkl'
PROCESSED_DATA_PATH = Path(__file__).parent.parent/'data'/'processed-data'


def train_and_save_tokenizer(vocab_size: int = 30522, min_frequency: int = 2):
    # ensure the tokenizer directory is empty
    if os.path.isdir(TOKENIZER_PATH):
        shutil.rmtree(TOKENIZER_PATH)
    os.makedirs(TOKENIZER_PATH)

    # get the train text
    train_files = sorted(list(map(str, (MLM_TARGET_DIR/'train').glob('*'))))

    # create, train a tokenizer on the train text data and save it
    byte_level_bpe_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    special_tokens = [
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>'
    ]
    byte_level_bpe_tokenizer.train(
        files=train_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens
    )