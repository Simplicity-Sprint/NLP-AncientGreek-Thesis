import os
import shutil

from pathlib import Path


PLAIN_TEXT_DATA_DIR = Path(__file__).parent.parent/'data'/'plain-text'
MLM_TARGET_DIR = (PLAIN_TEXT_DATA_DIR/'MLM').resolve()
POS_TARGET_DIR = (PLAIN_TEXT_DATA_DIR/'PoS').