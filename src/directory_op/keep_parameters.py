import os

base_dir = ""


# Delete all files in a directory, except file named pattern

def delete_files(path: str, pattern: str = "parameters.json"):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            if filename != pattern:
                os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_files(file_path)


if __name__ == "__main__":
    delete_files(base_dir)
