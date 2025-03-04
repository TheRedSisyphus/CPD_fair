from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import config.parameters as p
from config.logger import create_logger
from src.post_processing.FairnessDataset import FairnessDataset
from src.utils import parse_args, read_parameters, get_target, first_key, second_key

COLORS = ['b', 'r']


def plot_hist(fd: FairnessDataset, method_name: str, index: int, series: str, priv: bool):
    privileged = first_key(fd.map_sc) if priv else second_key(fd.map_sc)
    logger.debug(f"{priv}, {privileged}")
    # Filter only profiles changed
    condition = (fd.df['corrected_class'] != fd.df['predicted_class']) & (
            fd.df[fd.protected_attribute_name] == privileged)
    data = fd.df.loc[condition]
    if series == "cpd":
        data_to_plot = data["ratio_cpd_norm"]
    elif series == "scores":
        data_to_plot = fd.scores_fav.loc[condition]
    else:
        raise ValueError
    logger.debug(f"Data to plot for {method_name}, {series}, privileged : {priv}:\n{data_to_plot}")
    data_to_plot.hist(bins=np.linspace(0, 1, 30), alpha=0.6, label=method_name, color=COLORS[index])


def plot_all(fairness_data: list[tuple[str, FairnessDataset]], save_dir: Path | str, prefix: str) -> None:
    for series in ["cpd", "scores"]:
        for priv in [True, False]:
            fig, ax = plt.subplots()
            ax.grid()
            for i, (method_name, fd) in enumerate(fairness_data):
                plot_hist(fd, method_name, i, series, priv=priv)

            title = "CPD distributions of " if series == 'cpd' else "Model scores distributions of "
            title += "affected privileged profiles" if priv else "affected unprivileged profiles"
            ax.set_title(title, wrap=True)
            axis_x_name = r"Normalized difference $CPD_{fav} - CPD_{unfav}$" if series == 'cpd' else "Model scores for favored class"
            ax.set_xlabel(axis_x_name)
            ax.set(xlim=(0, 1))
            ax.set_ylabel("Number of profile changed")
            ax.legend()
            label_priv = 'priv' if priv else 'unpriv'
            fig.savefig(save_dir / f'plot_{prefix}_{series}_{label_priv}.pdf', dpi=300, format='pdf')


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
    params = read_parameters(args.param)

    # endregion

    fair_data_cpd = FairnessDataset(
        df=pd.read_csv(experiment_dir / "post_cpl.csv", index_col='inputId'),
        favorable_label=params["favorable_label"],
        unfavorable_label=params["unfavorable_label"],
        privileged_group=params["privileged_group"],
        unprivileged_group=params["unprivileged_group"],
        protected_attribute_name=params["pa_name"],
        target_name=get_target(params["db_name"]),
        dir_cpd=experiment_dir.parent / 'cpd')

    fair_data_to = FairnessDataset(
        df=pd.read_csv(experiment_dir / "threshop.csv", index_col='inputId'),
        favorable_label=params["favorable_label"],
        unfavorable_label=params["unfavorable_label"],
        privileged_group=params["privileged_group"],
        unprivileged_group=params["unprivileged_group"],
        protected_attribute_name=params["pa_name"],
        target_name=get_target(params["db_name"]),
        dir_cpd=experiment_dir.parent / 'cpd')

    fair_data_roc = FairnessDataset(
        df=pd.read_csv(experiment_dir / "roc.csv", index_col='inputId'),
        favorable_label=params["favorable_label"],
        unfavorable_label=params["unfavorable_label"],
        privileged_group=params["privileged_group"],
        unprivileged_group=params["unprivileged_group"],
        protected_attribute_name=params["pa_name"],
        target_name=get_target(params["db_name"]),
        dir_cpd=experiment_dir.parent / 'cpd')

    plot_all(fairness_data=[('CPD', fair_data_cpd), ('ROC', fair_data_roc)], save_dir=experiment_dir, prefix='ROC')
    plot_all(fairness_data=[('CPD', fair_data_cpd), ('TO', fair_data_to)], save_dir=experiment_dir, prefix='TO')
