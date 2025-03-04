import filecmp
from pathlib import Path

from PIL import Image


def is_dir_identical(dir1: Path, dir2: Path):
    """Compare all files recursively in two directories. Ignore log files"""
    for item in dir1.iterdir():
        if item.is_file() and item.suffix != '.log':
            if not (dir2 / item.name).exists():
                print(f"File {dir2 / item.name} doesn't exists")
            else:
                identical = filecmp.cmp(item, dir2 / item.name)
                if not identical:
                    print(f"File {item.stem} in {dir1} and in {dir2} are different")
        if item.is_dir():
            is_dir_identical(item, dir2 / item.name)


def is_plot_identical(dir1: Path, dir2: Path):
    """Compare all png files recursively in two directories"""
    for item in dir1.iterdir():
        if item.is_file() and item.suffix == '.png':
            if not (dir2 / item.name).exists():
                print(f"File {dir2 / item.name} doesn't exists")
            else:
                im1 = Image.open(item)
                im2 = Image.open(dir2 / item.name)
                identical = list(im1.getdata()) == list(im2.getdata())
                if not identical:
                    print(f"DIFFERENT : File {item.stem} in {dir1} and in {dir2}")
                else:
                    print('')
                    # print(f"IDENTICAL : File {item.stem} in {dir1} and in {dir2}")
        if item.is_dir():
            is_plot_identical(item, dir2 / item.name)


if __name__ == "__main__":
    is_plot_identical(Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments/"),
                      Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_or/"))
