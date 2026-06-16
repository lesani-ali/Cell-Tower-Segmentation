import logging
from pathlib import Path
from typing import Optional

import tqdm


class TqdmHandler(logging.StreamHandler):
    """Logging handler that writes through tqdm.write() to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:

    logger = logging.getLogger("seg_cell_tower")

    if logger.hasHandlers():
        logger.handlers.clear()

    base_level = (
        logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    )
    logger.setLevel(base_level)

    if verbose:
        # Detailed format for debugging
        frmt = (
            "%(levelname)-15s %(asctime)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    else:
        # Simple format for normal use
        frmt = "%(levelname)-15s %(asctime)s | %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    # --- Console Handler ---
    console_handler = TqdmHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt=frmt, datefmt=datefmt)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        plain_formatter = logging.Formatter(fmt=frmt, datefmt=datefmt)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"seg_cell_tower.{name}")
    return logging.getLogger("seg_cell_tower")