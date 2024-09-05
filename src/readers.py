import os
from typing import Optional

import numpy as np

import config.parameters as p
from config.logger import create_logger

logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)


def np_file_reader(path: str, header: Optional[tuple[str, ...]] = None) -> np.ndarray:
    """
    Read the file and returns the structured numpy array containing the file
    @param path: Path of the file
    @param header: Optional. If not None, used for verification.
    @return: Structured numpy array
    """
    array = np.genfromtxt(path, encoding='UTF-8', dtype=None, names=True)

    header_names = array.dtype.names

    if header is not None:
        if header_names != header:
            raise ValueError(
                f'ERROR file_reader: Header of file is not the one expected. Got {header} instead of {header_names}')

    if not array.shape:
        return array
    if array.shape[0] <= 0:
        logger.warning(f'Empty file at {path}')
    return array


def file_reader(path: str, sep: str = ' ', header: Optional[list[str]] = None) -> list[list[str]]:
    """
    @param path: Path of the file to read
    @param sep: separator used, default is ' '. For csv files it's ','
    @param header: Optional. If passed verify that the first line is the same as header.
    If header is [''], file reader skip the first line
    @return: List of list of string representing the file.
    """
    if not os.path.exists(path):
        raise ValueError(f'ERROR file_reader: Path {path} does not exist')
    with open(path, 'r') as file:
        content = file.readlines()

    if not content:
        logger.warning(f'Empty file at {path}')
        return []
    else:
        first_line = content[0]
        first_line = first_line.strip('\n').strip(' ').split(sep)
        result = []
        if header:
            if header != [''] and first_line != header:
                raise ValueError(
                    f'ERROR file_reader: Header of file is not the one expected. Got {first_line} instead of {header}')
        else:
            result.append(first_line)
        expected_len = len(first_line)

        if len(content) == 1:
            logger.warning(f'No content in file {path}')
            return result

        for line_nbr, line in enumerate(content[1:]):
            # remove \n and ' ' at the end and at the beginning of line, and make a list
            line = line.strip('\n').strip(' ').split(sep)
            # Ignore empty lines
            if line == [] or line == ['']:
                continue
            if len(line) != expected_len:
                raise ValueError(
                    f'ERROR file_reader: line {line_nbr} has incorrect format. Expected length {expected_len}, got {len(line)}')

            result.append(line)

        return result
