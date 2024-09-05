import os
from typing import Iterable

from config.parameters import project_path, exp_dir_path


def run_exps(exp_list: int | Iterable[int], result_list: int | Iterable[int], mode: str = 'all'):
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if mode not in ["CPDExtract", "lh_repr", "all"]:
        raise ValueError

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(result_list, int):
        result_list = [result_list]

    for exp in exp_list:
        print(f"Running exp {exp}")
        for res in result_list:
            print(f"Running result {res}")
            if mode != "lh_repr":
                os.system(
                    f"python3 src/pipelines/CPDExtract.py -p '{os.path.join(exp_dir_path, f"exp_{exp}/", f"results_{res}")}/parameters.json'")
            if mode != "CPDExtract":
                os.system(
                    f"python3 src/pipelines/lh_repr.py -p '{os.path.join(exp_dir_path,  f"exp_{exp}/", f"results_{res}")}/repr/parameters.json'")


if __name__ == "__main__":
    run_exps(exp_list=1,
             result_list=2,
             mode="lh_repr")
