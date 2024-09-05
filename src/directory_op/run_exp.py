import os
from typing import Iterable

from config.parameters import project_path, exp_dir_path, data_dir_name


def run_exps(exp_list: int | Iterable[int], result_list: int | Iterable[int], mode: str | Iterable[str] = 'all'):
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(result_list, int):
        result_list = [result_list]
    if not isinstance(mode, set):
        if mode == 'all':
            mode = {'prep_exp', 'CPDExtract', 'lh_repr'}
        else:
            mode = {mode}
    if not set(mode).issubset(['prep_exp', 'CPDExtract', 'lh_repr']):
        raise ValueError("mode arg must be a subset of ['prep_exp', 'CPDExtract', 'lh_repr']")

    for exp in exp_list:
        print(f"\n=== RUNNING EXP {exp} ===")
        print(f"Path : {os.path.join(exp_dir_path, f"exp_{exp}")}\n")
        if "prep_exp" in mode:
            return_code = os.system(
                f"python3 src/pipelines/prep_exp.py -p '{os.path.join(exp_dir_path, f"exp_{exp}", data_dir_name)}/parameters.json'")
            if return_code != 0:
                break
        for res in result_list:
            print(f"\n--- RUNNING RESULT {res} ---")
            print(f"Path : {os.path.join(exp_dir_path, f"exp_{exp}", f"results_{res}")}\n")
            if "CPDExtract" in mode:
                return_code = os.system(
                    f"python3 src/pipelines/CPDExtract.py -p '{os.path.join(exp_dir_path, f"exp_{exp}", f"results_{res}")}/parameters.json'")
                if return_code != 0:
                    break
            if "lh_repr" in mode:
                return_code = os.system(
                    f"python3 src/pipelines/lh_repr.py -p '{os.path.join(exp_dir_path, f"exp_{exp}", f"results_{res}")}/repr/parameters.json'")
                if return_code != 0:
                    break


if __name__ == "__main__":
    run_exps(exp_list=[2, 3, 4, 5, 6, 7],
             result_list=[1, 2],
             mode="lh_repr")
