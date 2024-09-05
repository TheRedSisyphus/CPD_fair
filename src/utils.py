import argparse
import os

import pandas as pd

import config.parameters as p
from config.logger import create_logger

logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)
ATTR_INFO = tuple[str, dict[float: pd.Series], dict[float: pd.Series]] | str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", required=True, type=str, help="Path to exp parameters")
    return parser.parse_args()


def cross_tab_percent(series_1: pd.Series, series_2: pd.Series):
    """Make a cross tab and display percent for two series"""
    return pd.crosstab(series_1, series_2, margins=True).apply(
        lambda row: row.apply(lambda val: f"{val} ({100 * val / len(series_1):.2f}%)"), axis=1)


def get_protec_attr(descr: str, data_path: str | None = None) -> ATTR_INFO:
    if data_path is not None:
        data = pd.read_csv(data_path)
        if descr.count("=") == 1:  # Binarized attribute
            attr, value = descr.split("=")
            return attr, {value: data[data[attr] == value]['inputId']}, {value: data[data[attr] != value]['inputId']}
        elif descr.count("/") == 1 and descr.count(":") == 1:  # Multiple attribute
            attr, values = descr.split(":")
            value_1, value_2 = descr.split("/")
            return attr, {value_1: data[data[attr] == value_1]['inputId']}, {
                value_2: data[data[attr] != value_2]['inputId']}
        else:
            raise ValueError("Invalid format in parameters file for protected attribute")
    else:
        if descr.count("=") == 1:
            attr, value = descr.split("=")
            return attr
        elif descr.count("/") == 1 and descr.count(":") == 1:
            attr, values = descr.split(":")
            return attr
        else:
            raise ValueError("Invalid format in parameters file for protected attribute")


# region Data related function : To modify if data get modified

def get_target(db_name: str) -> str:
    """Returns name of the target depending on db used"""
    if db_name == "adult":
        return "income"
    elif db_name == "credit":
        return "default payment next month"
    elif db_name == "compas":
        return "two_year_recid"
    else:
        raise ValueError("get_target : Unknown name for database")


def get_map_str(params: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """To have readable output class and sensitive attribute"""
    # region target
    if params['target'] == 'income':
        map_oc_str = {"0": "lr", "1": "hr"}
    elif params['target'] == "default payment next month":
        map_oc_str = {"0": "ndef", "1": "def"}
    elif params['target'] == "two_year_recid":
        map_oc_str = {"0": "nrecid", "1": "recid"}
    else:
        raise ValueError(f'Invalid parameters : got target "{params['target']}"')
    # endregion

    # region sensitive attribute
    if params['sens_attr'] == 'sex':
        map_sc_str = {"0": "f", "1": "m"}
    elif params['sens_attr'] == 'educ':
        map_sc_str = {"0": "le", "1": "he"}
    elif params['sens_attr'] == 'race':
        map_sc_str = {"0": "nwhite", "1": "white"}
    elif params['sens_attr'] == 'age':
        map_sc_str = {"0": "young", "1": "old"}
    elif params['sens_attr'] == 'race=African-American':
        map_sc_str = {"1": "afr_amer", "0": "n_afr_amer"}
    else:
        raise ValueError(f'Invalid parameters : got sensitive attribute "{params['sens_attr']}"')
    # endregion
    return map_sc_str, map_oc_str


# endregion

# region Post processing

class FairnessDataset:
    def __init__(self,
                 favorable_label,
                 unfavorable_label,
                 protected_attribute_name: str,
                 target_name: str,
                 scores: pd.Series,
                 path: str = None,
                 df: pd.DataFrame = None):
        super().__init__()
        if path is not None and df is not None:
            raise ValueError
        elif path is not None:
            self.df = pd.read_csv(path)
        elif df is not None:
            self.df = df
        self.scores = scores
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_attribute_name = protected_attribute_name
        self.target_name = target_name
        self.protected_attribute = self.df[protected_attribute_name]

# endregion
