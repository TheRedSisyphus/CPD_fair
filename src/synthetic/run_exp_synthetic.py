import os
from typing import Iterable

from config.parameters import project_path, exp_dir_path, data_dir_name


def run_exps(exp_list: int | Iterable[int], mode: str | Iterable[str] = 'all'):
    """
    Run experiments
    :param exp_list: List of experiments to run
    :param mode: Either 'prep_exp_synthetic', 'CPDExtract' or 'lh_repr_synthetic' to specify to specific part of the experiment.
    Use 'all' to run all
    """
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(mode, str):
        if mode == 'all':
            mode = {'prep_exp_synthetic', 'CPDExtract', 'lh_repr_synthetic'}
        else:
            mode = {mode}
    else:
        mode = set(mode)
    if not set(mode).issubset(['prep_exp_synthetic', 'CPDExtract', 'lh_repr_synthetic']):
        raise ValueError("mode arg must be a subset of ['prep_exp_synthetic', 'CPDExtract', 'lh_repr_synthetic']")

    for exp in exp_list:
        print(f"\n=== RUNNING EXP {exp} ===")
        exp_path = exp_dir_path / f"exp_{exp}"
        print(f"Path : {exp_path}")
        if "prep_exp_synthetic" in mode:
            return_code = os.system(
                f"python3 src/synthetic/prep_exp_synthetic.py -p '{exp_path / data_dir_name}/parameters.json'> '{exp_path / data_dir_name}/exp.log'")
            if return_code != 0:
                break
        for res in [1, 2]:
            print(f"\n--- RUNNING RESULT {res} ---")
            result_path = exp_path / f"results_{res}"
            print(f"Path : {result_path}\n")
            if "CPDExtract" in mode:
                return_code = os.system(
                    f"python3 src/pipelines/CPDExtract.py -p '{result_path}/parameters.json'> '{result_path}/exp.log'")
                if return_code != 0:
                    break
        if "lh_repr_synthetic" in mode:
            return_code = os.system(
                f"python3 src/synthetic/lh_repr_synthetic.py -p '{exp_path / "plot"}/parameters.json'> '{exp_path / "plot"}/exp.log'")
            if return_code != 0:
                break


if __name__ == "__main__":
    run_exps(exp_list=[20004], mode='all')
