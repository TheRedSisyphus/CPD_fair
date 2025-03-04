import os
from typing import Iterable

from config.parameters import project_path, exp_dir_path, data_dir_name


def run_exps(exp_list: int | Iterable[int], mode: str | Iterable[str] = 'all'):
    """
    Run experiments
    :param exp_list: List of experiments to run
    :param mode: Either 'prep_exp', 'extract_contribs' or 'lh_repr' to specify to specific part of the experiment.
    Use 'all' to run all
    """
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(mode, str):
        if mode == 'all':
            mode = {'prep_exp', 'extract_contribs', 'repr'}
        else:
            mode = {mode}
    else:
        mode = set(mode)
    if not set(mode).issubset(['prep_exp', 'extract_contribs', 'repr']):
        raise ValueError("mode arg must be a subset of ['prep_exp', 'extract_contribs', 'repr']")

    for exp in exp_list:
        print(f"\n=== RUNNING EXP {exp} ===")
        exp_path = exp_dir_path / "histograms" / f"exp_{exp}"
        print(f"Path : {exp_path}")
        if "prep_exp" in mode:
            return_code = os.system(
                f"python3 src/pipelines/prep_exp.py -p '{exp_path / data_dir_name}/parameters.json'> '{exp_path / data_dir_name}/exp.log'")
            if return_code != 0:
                break

        print(f"\n--- RUNNING CPD EXTRACT ---")
        result_path = exp_path / f"cpd"
        print(f"Path : {result_path}\n")
        if "extract_contribs" in mode:
            return_code = os.system(
                f"python3 src/plot_hist/extract_contribs.py -p '{result_path}/parameters.json'> '{result_path}/exp.log'")
            if return_code != 0:
                break
        if "repr" in mode:
            return_code = os.system(
                f"python3 src/plot_hist/plot_node_diagram.py -p '{result_path}/parameters.json'> '{result_path}/exp.log'")
            if return_code != 0:
                break


if __name__ == "__main__":
    run_exps(exp_list=[1, 2, 3], mode='all')
