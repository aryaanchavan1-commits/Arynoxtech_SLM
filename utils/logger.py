import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)