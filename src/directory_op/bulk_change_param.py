import json
import os
from typing import Callable

from config.parameters import project_path


def change_param(path: str, change_repr: bool, key_to_change: str, action_to_perform: Callable) -> None:
    """
    :param path: Path to the dir containing multiple experiments dir
    :param change_repr: Bool. If true, only change parameters inside repr directory.
    If false, only change parameters inside experiments dir (param inside repr are not affected)
    :param key_to_change: String
    :param action_to_perform: See below
    :return: Nothing, but modify parameters.json files found
    """
    path = os.path.join(project_path, path)
    parameters_path = None
    for exp_dir in os.listdir(path):
        exp_dir_path = os.path.join(path, exp_dir)
        if os.path.isdir(exp_dir_path):
            for filename in os.listdir(exp_dir_path):
                file_path = os.path.join(exp_dir_path, filename)
                if change_repr:
                    if os.path.isdir(file_path):
                        if filename == "repr":
                            for param_file in os.listdir(file_path):
                                if param_file == "parameters.json":
                                    parameters_path = os.path.join(file_path, param_file)
                else:
                    if os.path.isfile(file_path) and filename == "parameters.json":
                        parameters_path = file_path

            if parameters_path is None:
                raise ValueError(f"File parameters.json not found in {path}")

            with open(parameters_path) as f_params:
                params = json.load(f_params)
            action_to_perform(params, key_to_change)
            with open(parameters_path, 'w', encoding='utf8') as f_params:
                json.dump(params, f_params, ensure_ascii=False, indent=2)


# region Actions to perform
def remove_key(param: dict[str, str], key_to_change: str):
    """Remove a key that already exists. If it doesn't exist, does not raise any exception"""
    param.pop(key_to_change, None)


def replace_key_value(to_replace: str, replacement: str):
    """Replace portion of string if the value is a string"""
    return lambda p, k: p.update({k: p[k].replace(to_replace, replacement)})


def update_key(new_value):
    """To create a new key or updating an existing one"""
    return lambda p, k: p.update({k: new_value})


def swap_keys(key_to_swap):
    def swap(dic, key, swap_key):
        dic[key], dic[swap_key] = dic[swap_key], dic[key]

    return lambda p, k: swap(p, k, key_to_swap)


# endregion

if __name__ == "__main__":
    change_param(path="/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments/adult/adult_sex/blah",
                 change_repr=False,
                 key_to_change="",
                 action_to_perform=update_key(""))
