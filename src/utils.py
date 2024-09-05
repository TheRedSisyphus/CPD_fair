import argparse

import pandas as pd

ATTR_INFO = tuple[str, pd.Series] | str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", required=True, type=str, help="Path to exp parameters")
    return parser.parse_args()


def cross_tab_percent(series_1: pd.Series, series_2: pd.Series):
    """Make a cross tab and display percent for two series"""
    return pd.crosstab(series_1, series_2, margins=True).apply(
        lambda row: row.apply(lambda val: f"{val} ({100 * val / len(series_1):.2f}%)"), axis=1)


def get_protec_attr(descr: str, data_path: str | pd.DataFrame | None = None)->ATTR_INFO:
    if data_path is not None:
        if isinstance(data_path, str):
            data = pd.read_csv(data_path, index_col='inputId')
        else:
            data = data_path
        if descr.count("<") == 1:
            attr, value = descr.split("<")
            value = float(value)
            return attr, (data[attr] < value).squeeze().astype(int)
        elif descr.count("<=") == 1:
            attr, value = descr.split("<=")
            value = float(value)
            return attr, (data[attr] <= value).squeeze().astype(int)
        if descr.count(">") == 1:
            attr, value = descr.split(">")
            value = float(value)
            return attr, (data[attr] > value).squeeze().astype(int)
        elif descr.count(">=") == 1:
            attr, value = descr.split(">=")
            value = float(value)
            return attr, (data[attr] >= value).squeeze().astype(int)
        elif 0 < descr.count("=") <= 2:  # Binarized attribute
            attr, value = descr.rsplit('=', 1)
            value = float(value)
            return attr, (data[attr] == value).squeeze()
        elif descr.count("/") == 1 and descr.count(":") == 1:  # Multiple attribute
            attr, values = descr.split(":")
            value_1, value_2 = descr.split("/")
            value_1 = float(value_1)

            return attr, data[attr].map({value_1: 1, value_2: 0}).fillna(-1).astype(int)
        else:
            raise ValueError("Invalid format in parameters file for protected attribute")
    else:
        if descr.count("<") == 1:
            attr, _ = descr.split("<")
            return attr
        elif descr.count("<=") == 1:
            attr, _ = descr.split("<=")
            return attr
        if descr.count(">") == 1:
            attr, _ = descr.split(">")
            return attr
        elif descr.count(">=") == 1:
            attr, _ = descr.split(">=")
            return attr

        if 0 < descr.count("=") <= 2:
            attr, _ = descr.rsplit("=", 1)
            return attr
        elif descr.count("/") == 1 and descr.count(":") == 1:
            attr, _ = descr.split(":")
            return attr
        else:
            raise ValueError(f"Invalid format in parameters file for protected attribute. Got {descr}")


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
    elif params['sens_attr'] == 'race=White':
        map_sc_str = {"1": "white", "0": "n_white"}
    else:
        raise ValueError(f'Invalid parameters : got sensitive attribute "{params['sens_attr']}"')
    # endregion
    return map_sc_str, map_oc_str

# endregion
