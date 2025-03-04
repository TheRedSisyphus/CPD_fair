import json
import typing
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import tabulate
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score

from config import parameters as p
from config.logger import create_logger
from src.readers import file_reader
from src.utils import get_map_oc_str, get_map_sc_str, first_value, second_value, read_conditional_pa, normalize_series

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


def percent(array: list | int | float, denominator: int) -> list[str]:
    """Display percentage of array divided by denominator in a pretty way"""
    if isinstance(array, list):
        result = []
        for elem in array:
            perc = elem / denominator
            result.append(str(elem) + f" ({perc * 100:.2f}%)")
    elif isinstance(array, int) or isinstance(array, float):
        perc = array / denominator
        result = f"{perc * 100:.2f}%"
    else:
        raise ValueError
    return result


def parse_cpd_files(files_cpd_class_0: typing.Iterable[str | Path],
                    files_cpd_class_1: typing.Iterable[str | Path]) -> tuple[dict[int, float], dict[int, float]]:
    """Read and parse to python dict cpl data from two list of files"""

    # read CPL files
    content_class_0 = []
    content_class_1 = []
    for f in files_cpd_class_0:
        content_class_0 += file_reader(f, header=p.lh_header)
    for f in files_cpd_class_1:
        content_class_1 += file_reader(f, header=p.lh_header)

    # Parsing to python dict
    dict_class_0 = {}
    for row in content_class_0:
        if len(row) != 2:
            raise ValueError(f"CPL files invalid: {row}")
        index, cpd = row
        dict_class_0[int(index)] = float(cpd)

    dict_class_1 = {}
    for row in content_class_1:
        if len(row) != 2:
            raise ValueError(f"CPL files invalid: {row}")
        index, cpd = row
        dict_class_1[int(index)] = float(cpd)

    return dict_class_0, dict_class_1


def get_performance_criterion(method_name: str) -> typing.Callable:
    if method_name == "precision_score":
        return precision_score
    elif method_name == "accuracy_score":
        return accuracy_score
    elif method_name == "balanced_accuracy_score":
        return balanced_accuracy_score
    else:
        raise ValueError(f"Invalid method name for performance criterion, got {method_name}")


class FairnessDataset:
    """Wrapper for Pandas Dataframe, including notion of fairness"""

    def __init__(self,
                 favorable_label,
                 unfavorable_label,
                 privileged_group,
                 unprivileged_group,
                 protected_attribute_name: str,
                 target_name: str,
                 df: pd.DataFrame | str | Path,
                 dir_cpd: str | Path,
                 scores_fav: pd.Series | None = None,
                 scores_unfav: pd.Series | None = None):
        if df is None:
            raise ValueError
        else:
            if isinstance(df, pd.DataFrame):
                self.df = df
            elif isinstance(df, str | Path):
                self.df = pd.read_csv(df)
            else:
                raise ValueError(f"You must give a path or a Dataframe object, got {type(df)}")

        # If we considered conditional protected attribute
        if protected_attribute_name not in self.df.columns:
            _, _, study_group = read_conditional_pa(name_pa=protected_attribute_name, data=self.df)
            self.df[protected_attribute_name] = study_group
            self.protected_attribute_name = protected_attribute_name
        else:
            self.protected_attribute_name = protected_attribute_name

        self.target_name = target_name

        # In FairnessDataset, ground truth column is called so
        # !! It assumes that original dataset contains ground truth value in the column name 'target_name'
        if self.target_name in self.df.columns:
            self.df.rename(columns={self.target_name: "ground_truth"}, inplace=True)

        self.fav = favorable_label
        self.unfav = unfavorable_label

        self.priv = privileged_group
        self.unpriv = unprivileged_group

        self.scores_fav = scores_fav
        self.scores_unfav = scores_fav

        self.map_oc = get_map_oc_str(self.target_name)
        self.map_sc = get_map_sc_str(self.protected_attribute_name)

        self.dir_cpd = dir_cpd
        self.files_cpd_class_fav = [self.dir_cpd / Path(f'lh_{self.map_sc[self.priv]}-{self.map_oc[self.fav]}.csv'),
                                    self.dir_cpd / Path(f'lh_{self.map_sc[self.unpriv]}-{self.map_oc[self.fav]}.csv')]

        self.files_cpd_class_unfav = [self.dir_cpd / Path(f'lh_{self.map_sc[self.priv]}-{self.map_oc[self.unfav]}.csv'),
                                      self.dir_cpd / Path(
                                          f'lh_{self.map_sc[self.unpriv]}-{self.map_oc[self.unfav]}.csv')]

        if ("predicted_class" not in self.df.columns or
                "corrected_class" not in self.df.columns or
                "cpd_class_fav" not in self.df.columns or
                "cpd_class_unfav" not in self.df.columns or
                self.scores_unfav is None or
                self.scores_fav is None):
            self.initialize()

    def initialize(self):
        """Create needed column for fairness dataset. Should only be called once at initialization."""
        logger.debug("Initialize FairDf")
        dict_class_fav, dict_class_unfav = parse_cpd_files(self.files_cpd_class_fav, self.files_cpd_class_unfav)

        if self.scores_fav is not None and self.scores_unfav is not None:
            # Setting column to predictions made by model
            if self.fav == 0:  # Ex : scores_unfav = 0.25 -> pred = 0 -> fav / scores_unfav = 0.75 -> pred = 1 -> unfav
                self.df["model_scores"] = self.scores_unfav
            elif self.fav == 1:  # Ex : scores_fav = 0.25 -> pred = 0 -> unfav / scores_fav = 0.75 -> pred = 1 -> fav
                self.df["model_scores"] = self.scores_fav
            else:
                raise ValueError("Favorable and unfavorable label must be 0 or 1")
        else:
            if self.fav == 0:
                self.scores_fav = 1 - self.df["model_scores"]
                self.scores_unfav = self.df["model_scores"]
            elif self.fav == 1:
                self.scores_fav = self.df["model_scores"]
                self.scores_unfav = 1 - self.df["model_scores"]
            else:
                raise ValueError("Favorable and unfavorable label must be 0 or 1")

        model_prediction = self.df["model_scores"].round().astype(int)

        # Before fitting, corrected class is only predicted class
        if "predicted_class" not in self.df.columns:
            self.df["predicted_class"] = model_prediction
        if "corrected_class" not in self.df.columns:
            self.df["corrected_class"] = model_prediction

        # Adding cpl data to dataset_pred
        if "cpd_class_fav" not in self.df.columns:
            self.df["cpd_class_fav"] = pd.Series(dict_class_fav)

            if (self.df["cpd_class_fav"] == 0).any():
                raise ValueError("Found CPL values that are null (cpl from favored class)")

        if "cpd_class_unfav" not in self.df.columns:
            self.df["cpd_class_unfav"] = pd.Series(dict_class_unfav)

            if (self.df["cpd_class_unfav"] == 0).any():
                raise ValueError("Found CPL values that are null (cpl from unfavored class)")

        self.df["ratio_cpd_norm"] = normalize_series(self.df["cpd_class_fav"] - self.df["cpd_class_unfav"])

        logger.debug("Done")

    def create_from(self, df: pd.DataFrame) -> Self:
        """Return a new FairnessDataset object, with data from a Dataframe and other attribute identical to self"""
        return FairnessDataset(df=df, favorable_label=self.fav, unfavorable_label=self.unfav,
                               privileged_group=self.priv,
                               unprivileged_group=self.unpriv, protected_attribute_name=self.protected_attribute_name,
                               target_name=self.target_name, scores_fav=self.scores_fav, scores_unfav=self.scores_unfav,
                               dir_cpd=self.dir_cpd)

    def copy(self, deep: bool) -> Self:
        return FairnessDataset(df=self.df.copy(deep=deep), favorable_label=self.fav, unfavorable_label=self.unfav,
                               privileged_group=self.priv,
                               unprivileged_group=self.unpriv, protected_attribute_name=self.protected_attribute_name,
                               target_name=self.target_name, scores_fav=self.scores_fav, scores_unfav=self.scores_unfav,
                               dir_cpd=self.dir_cpd)

    def get_train_data(self, set_name_path: str | Path) -> Self:
        set_name = pd.read_csv(set_name_path, index_col='inputId')
        set_name = set_name.squeeze()
        train_data = self.df[set_name != 'test']
        train_fair_df = self.create_from(train_data)

        train_fair_df.scores_fav = train_fair_df.scores_fav[set_name != 'test']
        train_fair_df.scores_unfav = train_fair_df.scores_unfav[set_name != 'test']

        return train_fair_df

    # region Fairness metric

    def nbr_priv(self) -> int:
        """Return number of privileged inputs"""
        return self.df[self.df[self.protected_attribute_name] == self.priv].shape[0]

    def nbr_unpriv(self) -> int:
        """Return number of unprivileged inputs"""
        return self.df[self.df[self.protected_attribute_name] == self.unpriv].shape[0]

    def nbr_fav(self, result_column: str) -> int:
        """Return number of favorable inputs"""
        return self.df[self.df[result_column] == self.fav].shape[0]

    def nbr_unfav(self, result_column: str) -> int:
        """Return number of favorable inputs"""
        return self.df[self.df[result_column] == self.unfav].shape[0]

    def nbr_fav_priv(self, result_column: str) -> int:
        return (self.df[(self.df[result_column] == self.fav) &
                        (self.df[self.protected_attribute_name] == self.priv)].shape[0])

    def nbr_fav_unpriv(self, result_column: str) -> int:
        return (self.df[(self.df[result_column] == self.fav) &
                        (self.df[self.protected_attribute_name] == self.unpriv)].shape[0])

    def nbr_unfav_priv(self, result_column: str) -> int:
        return (self.df[(self.df[result_column] == self.unfav) &
                        (self.df[self.protected_attribute_name] == self.priv)].shape[0])

    def nbr_unfav_unpriv(self, result_column: str) -> int:
        return (self.df[(self.df[result_column] == self.unfav) &
                        (self.df[self.protected_attribute_name] == self.unpriv)].shape[0])

    def nbr_true_fav_priv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.fav) & (self.df["ground_truth"] == self.fav) & (
                self.df[self.protected_attribute_name] == self.priv)].shape[0]

    def nbr_true_fav_unpriv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.fav) & (self.df["ground_truth"] == self.fav) & (
                self.df[self.protected_attribute_name] == self.unpriv)].shape[0]

    def nbr_true_unfav_priv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.unfav) & (self.df["ground_truth"] == self.unfav) & (
                self.df[self.protected_attribute_name] == self.priv)].shape[0]

    def nbr_true_unfav_unpriv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.unfav) & (self.df["ground_truth"] == self.unfav) & (
                self.df[self.protected_attribute_name] == self.unpriv)].shape[0]

    def nbr_false_fav_priv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.fav) & (self.df["ground_truth"] == self.unfav) & (
                self.df[self.protected_attribute_name] == self.priv)].shape[0]

    def nbr_false_fav_unpriv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.fav) & (self.df["ground_truth"] == self.unfav) & (
                self.df[self.protected_attribute_name] == self.unpriv)].shape[0]

    def nbr_false_unfav_priv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.unfav) & (self.df["ground_truth"] == self.fav) & (
                self.df[self.protected_attribute_name] == self.priv)].shape[0]

    def nbr_false_unfav_unpriv(self, result_column: str) -> int:
        return self.df[(self.df[result_column] == self.unfav) & (self.df["ground_truth"] == self.fav) & (
                self.df[self.protected_attribute_name] == self.unpriv)].shape[0]

    # endregion

    # region fairness criterion
    def demo_parity_diff(self, result_column: str) -> float:
        """Return Demographic Parity Difference"""
        nbr_priv = self.nbr_priv()
        nbr_unpriv = self.nbr_unpriv()
        fav_priv = self.nbr_fav_priv(result_column)
        fav_unpriv = self.nbr_fav_unpriv(result_column)

        return fav_priv / nbr_priv - fav_unpriv / nbr_unpriv

    def equal_opportunity_diff(self, result_column: str) -> float:
        """Return Equal Opportunity Difference"""
        nbr_fav_priv = self.nbr_fav_priv("ground_truth")
        nbr_fav_unpriv = self.nbr_fav_unpriv("ground_truth")

        true_fav_priv = self.nbr_true_fav_priv(result_column)
        true_fav_unpriv = self.nbr_true_fav_unpriv(result_column)

        if nbr_fav_priv == 0 or nbr_fav_unpriv == 0:
            logger.warning("Undefined Equal Opportunity Difference")
            return p.POS_UNDEF_FLOAT

        return true_fav_priv / nbr_fav_priv - true_fav_unpriv / nbr_fav_unpriv

    def avg_odds_diff(self, result_column: str) -> float:
        """Return Average Odds Difference"""

        true_fav_priv = self.nbr_true_fav_priv(result_column)
        true_fav_unpriv = self.nbr_true_fav_unpriv(result_column)

        false_unfav_priv = self.nbr_false_unfav_priv(result_column)
        false_unfav_unpriv = self.nbr_false_unfav_unpriv(result_column)

        true_unfav_priv = self.nbr_true_unfav_priv(result_column)
        true_unfav_unpriv = self.nbr_true_unfav_unpriv(result_column)

        false_fav_priv = self.nbr_false_fav_priv(result_column)
        false_fav_unpriv = self.nbr_false_fav_unpriv(result_column)

        TPR_priv = true_fav_priv / (true_fav_priv + false_unfav_priv)
        TPR_unpriv = true_fav_unpriv / (true_fav_unpriv + false_unfav_unpriv)

        FPR_priv = false_fav_priv / (false_fav_priv + true_unfav_priv)
        FPR_unpriv = false_fav_unpriv / (false_fav_unpriv + true_unfav_unpriv)

        avg_odds_priv = (TPR_priv + FPR_priv) / 2
        avg_odds_unpriv = (TPR_unpriv + FPR_unpriv) / 2

        return avg_odds_priv - avg_odds_unpriv

    # endregion

    def describe(self) -> str:
        data_length = self.df.shape[0]

        count_pa = [self.nbr_priv(), self.nbr_unpriv()]

        count_pa_or_fav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df["ground_truth"] == self.fav)].shape[0]
            for pa in [self.priv, self.unpriv]]
        count_pa_or_unfav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df["ground_truth"] == self.unfav)].shape[0]
            for pa in [self.priv, self.unpriv]]

        count_pa_pred_fav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df['predicted_class'] == self.fav)].shape[0]
            for pa in [self.priv, self.unpriv]]
        count_pa_pred_unfav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df['predicted_class'] == self.unfav)].shape[
                0]
            for pa in [self.priv, self.unpriv]]

        count_pa_corr_fav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df['corrected_class'] == self.fav)].shape[0]
            for pa in [self.priv, self.unpriv]]
        count_pa_corr_unfav = [
            self.df[(self.df[self.protected_attribute_name] == pa) & (self.df['corrected_class'] == self.unfav)].shape[
                0]
            for pa in [self.priv, self.unpriv]]

        # Proportions
        proportions = [
            [f"{first_value(self.map_oc)} (ground truth)"] + percent(count_pa_or_fav + [sum(count_pa_or_fav)],
                                                                     data_length),
            [f"{second_value(self.map_oc)} (ground truth)"] + percent(count_pa_or_unfav + [sum(count_pa_or_unfav)],
                                                                      data_length),
            [],
            [f"{first_value(self.map_oc)} (predicted)"] + percent(count_pa_pred_fav + [sum(count_pa_pred_fav)],
                                                                  data_length),
            [f"{second_value(self.map_oc)} (predicted)"] + percent(count_pa_pred_unfav + [sum(count_pa_pred_unfav)],
                                                                   data_length),
            [],
            [f"{first_value(self.map_oc)} (corrected)"] + percent(count_pa_corr_fav + [sum(count_pa_corr_fav)],
                                                                  data_length),
            [f"{second_value(self.map_oc)} (corrected)"] + percent(count_pa_corr_unfav + [sum(count_pa_corr_unfav)],
                                                                   data_length),
            [],
            ["Total"] + percent(count_pa + [sum(count_pa)], data_length)
        ]

        description = f"{tabulate.tabulate(proportions, tablefmt='github', headers=[""] + [self.map_sc[self.priv],
                                                                                           self.map_sc[self.unpriv]] + ["Total"])}\n\n"

        # Number of status
        status_cond = [
            (self.df['predicted_class'] == self.fav) & (self.df['corrected_class'] == self.fav),
            (self.df['predicted_class'] == self.unfav) & (self.df['corrected_class'] == self.unfav),
            (self.df['predicted_class'] == self.fav) & (self.df['corrected_class'] == self.unfav),
            (self.df['predicted_class'] == self.unfav) & (self.df['corrected_class'] == self.fav)]
        status_choice = ['pos', 'neg', 'changedToNeg', 'changedToPos']
        status_col = pd.Series(np.select(status_cond, status_choice, default='unset'), index=self.df.index)
        status_count = status_col.value_counts().to_dict()

        description += (f"Number of each status is:\n"
                        f"neg: {status_count.get("neg")} (including {self.df[(self.df[self.protected_attribute_name] == self.priv) & (status_col == 'neg')].shape[0]} {first_value(self.map_sc)} and {self.df[(self.df[self.protected_attribute_name] == self.unpriv) & (status_col == 'neg')].shape[0]} {second_value(self.map_sc)})\n"
                        f"pos: {status_count.get("pos")} (including {self.df[(self.df[self.protected_attribute_name] == self.priv) & (status_col == 'pos')].shape[0]} {first_value(self.map_sc)} and {self.df[(self.df[self.protected_attribute_name] == self.unpriv) & (status_col == 'pos')].shape[0]} {second_value(self.map_sc)})\n"
                        f"changedToNeg: {status_count.get("changedToNeg")} (including {self.df[(self.df[self.protected_attribute_name] == self.priv) & (status_col == 'changedToNeg')].shape[0]} {first_value(self.map_sc)} and {self.df[(self.df[self.protected_attribute_name] == self.unpriv) & (status_col == 'changedToNeg')].shape[0]} {second_value(self.map_sc)})\n"
                        f"changedToPos: {status_count.get("changedToPos")} (including {self.df[(self.df[self.protected_attribute_name] == self.priv) & (status_col == 'changedToPos')].shape[0]} {first_value(self.map_sc)} and {self.df[(self.df[self.protected_attribute_name] == self.unpriv) & (status_col == 'changedToPos')].shape[0]} {second_value(self.map_sc)})\n")

        # Model performance after correction
        correct_corrections = self.df[self.df["ground_truth"] == self.df["corrected_class"]].shape[0]
        description += (f"\nModel accuracy after correction is:\n"
                        f"all: {percent(correct_corrections, data_length)}")

        description += "".join([
            f"\n{self.map_sc[pa]}: {percent(self.df[(self.df["ground_truth"] == self.df["corrected_class"]) & (self.df[self.protected_attribute_name] == pa)].shape[0], self.df[self.df[self.protected_attribute_name] == pa].shape[0])}"
            for pa in [self.priv, self.unpriv]]) + "\n\n"

        # Fairness criterion after correction
        description += f"Demographic Parity difference : {self.demo_parity_diff("corrected_class")}\n"
        description += f"Equal Opportunity difference : {self.equal_opportunity_diff("corrected_class")}\n"
        description += f"Average Odds difference : {self.avg_odds_diff("corrected_class")}\n"

        return description

    def compare(self, other, method_name: str, other_method_name: str) -> str:
        comparison = ""

        class_fav, class_unfav = first_value(self.map_oc), second_value(self.map_oc)

        changed_inputs = self.df[self.df["corrected_class"] != self.df["predicted_class"]]
        changed_inputs_other = other.df[other.df["corrected_class"] != other.df["predicted_class"]]

        common_id = changed_inputs.index.intersection(changed_inputs_other.index)
        not_in_common = changed_inputs.index.difference(changed_inputs_other.index)
        not_in_common_other = changed_inputs_other.index.difference(changed_inputs.index)

        common = self.df.iloc[common_id]
        not_in_common = self.df.iloc[not_in_common]
        not_in_common_other = self.df.iloc[not_in_common_other]

        comparison += f"\n{common.shape[0]} profiles in common, including: "
        for pa in [self.priv, self.unpriv]:
            comparison += f"{common[common[self.protected_attribute_name] == pa].shape[0]} {self.map_sc[pa]} and "
        comparison = comparison[:-5]  # remove "and"

        if common.shape[0] > 0:
            for pa in [self.priv, self.unpriv]:
                cpl_fav = common[common[self.protected_attribute_name] == pa]["cpd_class_fav"]
                cpl_unfav = common[common[self.protected_attribute_name] == pa]["cpd_class_unfav"]
                comparison += f"\n{self.map_sc[pa]}:\n"
                comparison += f"AVG_CPL_CLASS_{class_fav}: {cpl_fav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_fav}: {cpl_fav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_fav}: {cpl_fav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_fav}: {cpl_fav.max():.5f}\n"
                comparison += f"AVG_CPL_CLASS_{class_unfav}: {cpl_unfav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_unfav}: {cpl_unfav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_unfav}: {cpl_unfav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_unfav}: {cpl_unfav.max():.5f}\n"

        comparison += f"\n{not_in_common.shape[0]} profiles changed by {method_name} and not {other_method_name}, including: "
        for pa in [self.priv, self.unpriv]:
            comparison += f"{not_in_common[not_in_common[self.protected_attribute_name] == pa].shape[0]} {self.map_sc[pa]} and "
        comparison = comparison[:-5]  # remove "and"

        if not_in_common.shape[0] > 0:
            for pa in [self.priv, self.unpriv]:
                cpl_fav = not_in_common[not_in_common[self.protected_attribute_name] == pa]["cpd_class_fav"]
                cpl_unfav = not_in_common[not_in_common[self.protected_attribute_name] == pa]["cpd_class_unfav"]
                comparison += f"\n{self.map_sc[pa]}:\n"
                comparison += f"AVG_CPL_CLASS_{class_fav}: {cpl_fav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_fav}: {cpl_fav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_fav}: {cpl_fav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_fav}: {cpl_fav.max():.5f}\n"
                comparison += f"AVG_CPL_CLASS_{class_unfav}: {cpl_unfav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_unfav}: {cpl_unfav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_unfav}: {cpl_unfav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_unfav}: {cpl_unfav.max():.5f}\n"

        comparison += f"\n{not_in_common_other.shape[0]} profiles changed by {other_method_name} and not {method_name}, including: "
        for pa in [self.priv, self.unpriv]:
            comparison += f"{not_in_common_other[not_in_common_other[self.protected_attribute_name] == pa].shape[0]} {self.map_sc[pa]} and "
        comparison = comparison[:-5]  # remove "and"

        if not_in_common_other.shape[0] > 0:
            for pa in [self.priv, self.unpriv]:
                cpl_fav = not_in_common_other[not_in_common_other[self.protected_attribute_name] == pa]["cpd_class_fav"]
                cpl_unfav = not_in_common_other[not_in_common_other[self.protected_attribute_name] == pa][
                    "cpd_class_unfav"]
                comparison += f"\n{self.map_sc[pa]}:\n"
                comparison += f"AVG_CPL_CLASS_{class_fav}: {cpl_fav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_fav}: {cpl_fav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_fav}: {cpl_fav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_fav}: {cpl_fav.max():.5f}\n"
                comparison += f"AVG_CPL_CLASS_{class_unfav}: {cpl_unfav.mean():.5f}\n"
                comparison += f"STD_CPL_CLASS_{class_unfav}: {cpl_unfav.std():.5f}\n"
                comparison += f"MIN_CPL_CLASS_{class_unfav}: {cpl_unfav.min():.5f}\n"
                comparison += f"MAX_CPL_CLASS_{class_unfav}: {cpl_unfav.max():.5f}\n"

        return comparison

    def to_json_array(self, save_path: str | Path) -> None:

        """Export post-processing results to json array. The columns are:
        ID: input ID
        SENS_ATTR: Protected attribute of the input, format str
        ORIGINAL_CLASS: Ground Truth target of the input, format str
        PREDICTED_CLASS: Class of the input predicted by the model, format str
        CORRECTED_CLASS: Class of the input corrected by post-processing method, format str
        STATUS: "pos", "neg" (as pred by the model)
        "changedToPos", "changedToNeg" if post-processing correction changes model decision, format str
        MODEL_SCORE: Model score associated with the input
        CPD_SCORE: Normalized CPD score associated with the input
        """
        if self.scores_fav is None or self.scores_unfav is None:
            raise ValueError("Model scores must be defined for JSON export")

        content = [
            ["ID", "SENS_ATTR", "ORIGINAL_CLASS", "PREDICTED_CLASS", "CORRECTED_CLASS", "STATUS", "MODEL_SCORE",
             f"CPD_SCORE_{first_value(self.map_oc)}", f"CPD_SCORE_{second_value(self.map_oc)}"]
        ]
        dict_class_fav, dict_class_unfav = parse_cpd_files(self.files_cpd_class_fav, self.files_cpd_class_unfav)

        index: int  # Specifying that index is int
        for index, row in self.df.iterrows():
            prot_attr = int(row[self.protected_attribute_name])
            or_class = int(row['ground_truth'])
            pred_class = int(row['predicted_class'])
            corr_class = int(row['corrected_class'])
            model_score_fav = float(self.scores_fav[index])
            model_score_unfav = float(self.scores_unfav[index])
            if abs(1 - (model_score_fav + model_score_unfav)) > p.EPSILON:
                raise ValueError(
                    f"Model score do not sum to 1. Got {model_score_fav} + {model_score_unfav} = {model_score_fav + model_score_unfav}")

            cpd_score_class_fav = dict_class_fav[index]
            cpd_score_class_unfav = dict_class_unfav[index]

            if pred_class == 0 and corr_class == 0:
                status = 'neg'
            elif pred_class == 0 and corr_class == 1:
                status = 'changedToPos'
            elif pred_class == 1 and corr_class == 1:
                status = 'pos'
            elif pred_class == 1 and corr_class == 0:
                status = 'changedToNeg'
            else:
                raise ValueError

            new_row = [index,  # Index of input
                       self.map_sc[prot_attr],  # Protected attribute
                       self.map_oc[or_class],  # ground truth class
                       self.map_oc[pred_class],  # class predicted by the model
                       self.map_oc[corr_class],  # class corrected by post-processing method
                       status,  # Status as defined above
                       model_score_fav,  # Score returned by model for favored class
                       model_score_unfav,  # Score returned by model for unfavored class
                       cpd_score_class_fav,  # CPL of favored class
                       cpd_score_class_unfav  # CPL of unfavored class
                       ]

            content += [new_row]

        content[1:] = sorted(content[1:], key=lambda row_: row_[0])  # Sort by index
        with open(save_path.with_suffix('.json'), 'w') as file:
            file.write(json.dumps(content).replace('], [', '],\n['))
