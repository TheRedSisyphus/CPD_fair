import os
from typing import Iterable

from config.parameters import project_path, post_exp_dir_path, data_dir_name


def run_post(exp_list: int | Iterable[int], mode: str | Iterable[str] = 'all'):
    """Select the experiment number and run it. The parameters file is inside the directory. Similar to run_exps"""
    os.system(f'export PYTHONPATH="{project_path}"')
    os.chdir(project_path)

    if isinstance(exp_list, int):
        exp_list = [exp_list]
    if isinstance(mode, str):
        if mode == 'all':
            mode = {'CPDExtract', 'roc', 'threshop', 'post_cpl', 'compare', 'plot'}
        else:
            mode = {mode}
    else:
        mode = set(mode)
    if not set(mode).issubset(['CPDExtract','train_model', 'roc', 'threshop', 'post_cpl', 'compare', 'plot']):
        raise ValueError(f"mode arg must be a subset of ['CPDExtract', 'roc', 'threshop', 'post_cpl', 'compare']")

    for exp in exp_list:
        print(f"\n=== RUNNING EXP {exp} ===")
        exp_path = post_exp_dir_path / f"exp_{exp}"
        print(f"Path : {exp_path}")

        if "train_model" in mode:
            print(f"Preprocessing data and training model...")
            return_code = os.system(
                f"python3 src/pipelines/prep_exp.py -p '{exp_path / data_dir_name}/parameters.json'> '{exp_path / data_dir_name}/exp.log'")
            if return_code != 0:
                break

        if "CPDExtract" in mode:
            # 1. Preprocess data and train model on it
            print(f"Preprocessing data and training model...")
            return_code = os.system(
                f"python3 src/pipelines/prep_exp.py -p '{exp_path / data_dir_name}/parameters.json'> '{exp_path / data_dir_name}/exp.log'")
            if return_code != 0:
                break
            # 2. Retrieve CPL and normalize it
            cpd_path = exp_path / f"cpd"
            print(f"Retrieving cpl...")
            return_code = os.system(
                f"python3 src/pipelines/CPDExtract.py -p '{cpd_path}/parameters_1.json'> '{cpd_path}/exp_1.log'")
            if return_code != 0:
                break
            return_code = os.system(
                f"python3 src/pipelines/CPDExtract.py -p '{cpd_path}/parameters_2.json'> '{cpd_path}/exp_2.log'")
            if return_code != 0:
                break

        # 3. Apply the three post-processing methods and get a json array as a result
        post_path = exp_path / f"post"
        if "roc" in mode:
            print(f"Applying ROC...")
            return_code = os.system(
                f"python3 src/post_processing/pipelines/roc.py -p '{post_path}/parameters.json'> '{post_path}/roc.log'")
            if return_code != 0:
                break

        if "threshop" in mode:
            print(f"Applying Threshop...")
            return_code = os.system(
                f"python3 src/post_processing/pipelines/thresh_op.py -p '{post_path}/parameters.json'> '{post_path}/threshop.log'")
            if return_code != 0:
                break

        if "post_cpl" in mode:
            print(f"Applying Post CPL...")
            return_code = os.system(
                f"python3 src/post_processing/pipelines/post_cpl.py -p '{post_path}/parameters.json'> '{post_path}/post_cpl.log'")
            if return_code != 0:
                break

        # 4. Compare all the json array
        if "compare" in mode:
            print(f"Comparing post methods...")
            return_code = os.system(
                f"python3 src/post_processing/pipelines/compare_post.py -p '{post_path}/parameters.json'> '{post_path}/compare.log'")
            if return_code != 0:
                break

        if "plot" in mode:
            print("Plotting distributions...")
            return_code = os.system(
                f"python3 src/post_processing/pipelines/plot_post.py -p '{post_path}/parameters.json' > '{post_path}/plot.log'")
            if return_code != 0:
                break


if __name__ == "__main__":
    ALL_MODES = ['CPDExtract', 'roc', 'threshop', 'post_cpl', 'compare', 'plot']
    ALL_EXPS = [1, 2, 3, 11, 12, 13, 21, 22, 23, 101, 102, 103, 111, 112, 113, 201, 202, 203, 211, 212, 213, 301, 302,
                303, 311, 312, 313]

    run_post(exp_list=[101],
             mode=['threshop'])
