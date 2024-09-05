import logging
import os
import sys

from config.parameters import LOG_FORMAT


def create_logger(name: str = __name__, level: str = 'warning', file_dir: str | None = None) -> logging.Logger:
    logging_levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'warn': logging.WARNING,
        'info': logging.INFO,
        'test': logging.INFO,
        'debug': logging.DEBUG
    }

    # Create a custom logger
    logger = logging.getLogger(name)
    if file_dir is None:
        hand = logging.StreamHandler(sys.stdout)
    else:
        log_file = os.path.join(file_dir, 'exp.log')
        try:  # remove log file if it already exists
            os.remove(log_file)
        except OSError:
            pass
        hand = logging.FileHandler(log_file)

    # Create formatters
    basic_format = logging.Formatter(LOG_FORMAT)
    hand.setFormatter(basic_format)

    logger.addHandler(hand)

    logger.setLevel(logging_levels[level])
    return logger
