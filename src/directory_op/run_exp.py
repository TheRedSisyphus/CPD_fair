import os
from typing import Iterable

from config.parameters import project_path, exp_dir_path, data_dir_name


def run_exps(exp_list: int | Iterable[int], mode: str | Iterable[str] = 'all'):
    """
    Run experiments
    :param exp_list: List of experiments to run
    :param mode: Either 'prep_exp', 'CPDExtract' or 'lh_repr' to specify to specific part of the experiment.
    Use 'all' to run all
    """
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(mode, str):
        if mode == 'all':
            mode = {'prep_exp', 'CPDExtract', 'lh_repr'}
        else:
            mode = {mode}
    else:
        mode = set(mode)
    if not set(mode).issubset(['prep_exp', 'CPDExtract', 'lh_repr']):
        raise ValueError("mode arg must be a subset of ['prep_exp', 'CPDExtract', 'lh_repr']")

    for exp in exp_list:
        print(f"\n=== RUNNING EXP {exp} ===")
        exp_path = exp_dir_path / f"exp_{exp}"
        print(f"Path : {exp_path}")
        if "prep_exp" in mode:
            return_code = os.system(
                f"python3 src/pipelines/prep_exp.py -p '{exp_path / data_dir_name}/parameters.json'> '{exp_path / data_dir_name}/exp.log'")
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
            if "lh_repr" in mode:
                return_code = os.system(
                    f"python3 src/pipelines/lh_repr.py -p '{result_path}/repr/parameters.json'> '{result_path}/repr/exp.log'")
                if return_code != 0:
                    break


if __name__ == "__main__":
    # run_exps(
    #     exp_list=[6010, 6011, 6012, 6016, 6017, 6018, 6022, 6023, '6010_1', '6010_2', '6010_3', 8010, 8011, 8012, 8014,
    #               8015, 8016, 9010, 9011, 9012, 9013, 9014, 9015],
    #     mode='lh_repr')
    run_exps(
        exp_list=[5000],
        mode='CPDExtract')
