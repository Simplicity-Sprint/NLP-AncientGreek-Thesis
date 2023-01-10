import os
import shutil

from pathlib import Path


PLAIN_TEXT_DATA_DIR = Path(__file__).parent.parent/'data'/'plain-text'
MLM_TARGET_DIR = (PLAIN_TEXT_DATA_DIR/'MLM').resolve()
POS_TARGET_DIR = (PLAIN_TEXT_DATA_DIR/'PoS').resolve()


def create_dir_structure(dirpath: Path) -> None:
    """Deletes directory if it exists, creates it and adds trai