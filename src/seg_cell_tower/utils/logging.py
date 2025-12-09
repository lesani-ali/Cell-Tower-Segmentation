
import logging
import sys
import os


def setup_logger(log_path: str) -> None:

    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(f'{log_path}/log_file.log', 'w', 'utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)