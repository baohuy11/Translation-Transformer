import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "runs/__init__.py",
    "weights/__init__.py",
    "src/__init__.py",
    "src/dataset.py",
    "src/model.py",
    "src/config.py",
    "src/train.py",
    "src/translate.py",
    ".env",
    "setup.py",
    "app.py",
    "requirements.txt",
    "research/trials.ipynb",
    "research/attention_visual.ipynb",
    "test.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")