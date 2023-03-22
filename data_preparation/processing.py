
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
    byte_level_bpe_tokenizer.save_model(str(TOKENIZER_PATH))
    byte_level_bpe_tokenizer.save(str(TOKENIZER_PATH/'config.json'))


def convert_data_to_mlm_format(max_length: int = 512):
    # ensure the processed MLM data directory is empty
    if os.path.isdir(PROCESSED_DATA_PATH/'MLM'):
        shutil.rmtree(PROCESSED_DATA_PATH/'MLM')
    os.makedirs(PROCESSED_DATA_PATH/'MLM')

    # load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)

    def encode_sentences_of_file(filepath: Path) -> List[List[int]]:
        # read the text of the file
        with open(filepath, 'r') as fp:
            contents = fp.read()

        # documents are separated by two newlines when saved
        documents = re.split(r'\n{2,}', contents)

        # add chunks of data of maximum length 512 (including bos/eos tokens)
        data = []
        current = [tokenizer.bos_token_id]

        # encode the sentences of each document
        loop = tqdm(documents, desc=f'Converting {filepath} to input IDs')
        for document in loop:
            # sentences are separated by a newline when saved
            sentences = document.split('\n')

            # for each sentence, add it to the current segment if it fits,
            #  else end the current segment and create a new one with it
            for sentence in sentences:
                # skip bos/eos tokens as they are added manually
                encoded = tokenizer.encode(sentence, max_length=max_length,
                                           truncation=True)[1:-1]
                if len(current) + len(encoded) + 1 > max_length:
                    data.append(current + [tokenizer.eos_token_id])
                    current = [tokenizer.bos_token_id] + encoded
                else:
                    current += encoded

            # create a new segment from the last sentences of the document
            if len(current) > 1:
                data.append(current + [tokenizer.eos_token_id])

        return data

    # get all the files
    train_files = (MLM_TARGET_DIR/'train').glob('*')
    val_files = (MLM_TARGET_DIR/'val').glob('*')
    test_files = (MLM_TARGET_DIR/'test').glob('*')

    def transform_and_save_texts(files: List[Path], prefix: str) -> None:
        data = []
        for file in files: