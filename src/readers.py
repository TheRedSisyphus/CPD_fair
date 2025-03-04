from pathlib import Path
from typing import Optional

import numpy as np

import config.parameters as p
from config.logger import create_logger

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


def file_reader(path: Path, sep: str = ',', header: Optional[list[str]] = None) -> list[list[str]]:
    """
    :param path: Path of the file to read
    :param sep: separator used, default is ','.
    :param header: Optional. If passed verify that the first line is the same as header.
    If header is [''], file reader skip the first line
    :return: List of list of string representing the file.
    """
    if not path.exists():
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
