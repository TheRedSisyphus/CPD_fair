import os
import shutil
from pathlib import Path


def delete_files(path: Path, pattern: str = ".json"):
    """Delete recursively all files in a directory, except file named pattern"""
    for filename in path.iterdir():
        file_path = Path(path) / filename
        if file_path.is_file():
            if filename.suffix != pattern:
                os.remove(file_path)
        elif file_path.is_dir():
            delete_files(file_path)


def copy_dir(source: Path | list[Path], destination: Path | list[Path]):
    """Copy directory from source to destination, but only parameters.json files."""
    if isinstance(source, Path):
        source = [source]
    if isinstance(destination, Path):
        destination = [destination]
    if len(source) != len(destination):
        raise ValueError

    for index, s in enumerate(source):
        d = destination[index]
        if d.is_dir():
            raise ValueError(f"Directory {destination} already exists")
        shutil.copytree(src=s, dst=d,
                        ignore=shutil.ignore_patterns('*.txt',
                                                      '*.png',
                                                      '*.pdf',
                                                      '*.csv',
                                                      '*.pt',
                                                      'coordinates.json',
                                                      '*.log'))


if __name__ == "__main__":
    # delete_files(Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments/exp_9112"))
    copy_dir(source=Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments/histograms/exp_2"),
             destination=Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments/histograms/exp_3"))
