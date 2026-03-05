"""
Centralised logging setup.
Import `get_logger(__name__)` in every module.
"""

import logging
import sys
from core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    logger.propagate = False
    return logger
