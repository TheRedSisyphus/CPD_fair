import logging
import sys

from config.parameters import LOG_FORMAT

logging_levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'info': logging.INFO,
    'test': logging.DEBUG,
    'debug': logging.DEBUG
}


def create_logger(name, level: str = 'warning', log_format=None) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(name)
    hand = logging.StreamHandler(sys.stdout)

    # Create formatters
    if log_format is not None:
        basic_format = logging.Formatter(log_format)
    else:
        basic_format = logging.Formatter(LOG_FORMAT)
    hand.setFormatter(basic_format)

    logger.addHandler(hand)

    logger.setLevel(logging_levels[level])
    return logger
