import os
import shutil


# Delete all files in a directory, except file named pattern

def delete_files(path: str, pattern: str = "parameters.json"):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            if filename != pattern:
                os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_files(file_path)


def copy_dir(source: str, destination: str):
    if os.path.isdir(destination):
        raise ValueError(f"Directory {destination} already exists")
    shutil.copytree(src=source, dst=destination,
                    ignore=shutil.ignore_patterns('*.txt',
                                                  '*.png',
                                                  '*.csv',
                                                  '*.pt',
                                                  'coordinates.json',
                                                  '*.log'))


if __name__ == "__main__":
    copy_dir(source="",
             destination="")
