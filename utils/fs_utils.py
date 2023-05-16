import os
import shutil

from pathlib import Path


def delete_contents_of_directory(dir_path: Path) -> None:
    """Deletes all the contents (recursively) inside the given directory."""
    if not os.path.isdir(dir_path):
        return
    for file_path in dir_path.iterdir():
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    