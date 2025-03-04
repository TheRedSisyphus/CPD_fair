import argparse
import json
import math
import numbers
import random
import typing
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset

from config import parameters as p
from config.logger import create_logger

ATTR_INFO = tuple[str, pd.Series] | str
logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


def read_conditional_pa(name_pa: str, data: pd.DataFrame) -> tuple[str, float | None, pd.Series]:
    if name_pa.count("<=") == 1:
        attr, value = name_pa.split("<=")
        value = float(value)
        return attr, value, (data[attr] <= value).squeeze().astype(int)
    elif name_pa.count(">=") == 1:
        attr, value = name_pa.split(">=")
        value = float(value)
        return attr, value, (data[attr] >= value).squeeze().astype(int)

    elif name_pa.count("<") == 1:
        attr, value = name_pa.split("<")
        value = float(value)
        return attr, value, (data[attr] < value).squeeze().astype(int)
    elif name_pa.count(">") == 1:
        attr, value = name_pa.split(">")
        value = float(value)
        return attr, value, (data[attr] > value).squeeze().astype(int)

    elif name_pa.count("=") == 1:
        attr, value = name_pa.split("=")
        value = float(value)
        return attr, value, (data[attr] == value).squeeze().astype(int)
    else:
        if name_pa in data.columns:
            return name_pa, None, data[name_pa].squeeze().astype(int)
        else:
            raise ValueError(f"Column {name_pa} not found in dataframe")


def filter_indexes(indexes_array: list[list[str]],
                   set_name: str | None = None,
                   correct: str | bool | None = None,
                   output_class: str | list[str] | None = None,
                   sensitive_class: str | list[str] | None = None) -> list[int]:
    """
    :param indexes_array: Unfiltered indexes to read
    :param set_name: None, '', 'train', 'test', 'valid'. The name of the set to filter
    :param correct: None, '', 'true',True, 'false', False. Filter input where model prediction is correct/incorrect
    :param output_class: Output class to filter. If output_class is a list, filter all input that have oc in this list
    :param sensitive_class: Sensitive class to filter.
    If sensitive_class is a list, filter all input that have sc in this list
    :return: A list of ID of input filtered
    """
    filtered = indexes_array.copy()
    if set_name != '' and set_name is not None:
        filtered = [line for line in filtered if line[p.set_name_pos] == set_name]
    if correct in [True, 'true']:
        filtered = [line for line in filtered if line[p.true_class_pos] == line[p.pred_class_pos]]
    elif correct in [False, 'false']:
        filtered = [line for line in filtered if line[p.true_class_pos] != line[p.pred_class_pos]]

    if isinstance(output_class, str):
        output_class = [output_class]
    if isinstance(output_class, list):
        filtered = [line for line in filtered if line[p.true_class_pos] in output_class]
    if isinstance(sensitive_class, str):
        sensitive_class = [sensitive_class]
    if isinstance(sensitive_class, list):
        filtered = [line for line in filtered if line[p.sens_attr_pos] in sensitive_class]

    if len(filtered) <= 0:
        logger.warning(
            f'No indexes after filtering{" for " + set_name + " set" if set_name else ""},{" output class : " + str(output_class) if output_class else ""}{" sensitive class : " + str(sensitive_class) if sensitive_class else ""}')
        return []

    return [int(line[p.input_id_pos]) for line in filtered]


def filter_dataframe(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Filter inplace a pandas dataframe with filters. Doesn't apply filter if filter is invalid"""
    for key, value in filters.items():
        if key in df.columns:
            if value is not None:  # It is possible to pass a none value (no filtering is applied)
                if isinstance(value, typing.Iterable):
                    for v in value:
                        df = df[df[key] == v]
                elif isinstance(value, numbers.Number):
                    df = df[df[key] == value]
                else:
                    logger.warning(
                        f'get_data_loader : filter {key} = {value} not applied. {value} must be iterable or number')

        else:
            logger.warning(
                f'get_data_loader : filter {key} = {value} not applied. key not in dataframe column')

    return df


def dataframe_to_loader(df: DataFrame, target: str) -> DataLoader:
    """Convert pandas dataframe to dataloader. Remove inputId column before processing"""
    features = torch.tensor(df.drop([target], axis=1).values)
    target_values = torch.tensor(df[target].values)
    dataset = TensorDataset(features, target_values)
    data_loader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False)
    return data_loader


def get_data_loaders(df: pd.DataFrame, set_name: pd.Series, target: str) -> dict[str, DataLoader]:
    """From a pandas Dataframe, returns a dict with key 'all', 'train', 'test' and 'valid'
    and value torch Dataloaders corresponding the 'set'"""
    loaders = {}
    filtered_df = df
    filtered_set_name = set_name

    for name_of_set in [None, 'train', 'test', 'valid']:
        if name_of_set is None:
            loaders['all'] = dataframe_to_loader(df=filtered_df, target=target)
        else:
            condition = filtered_set_name == name_of_set  # Boolean series : True when set_name is the one wanted
            condition = condition[condition]  # Keep only row where it's true
            filtered_df_sn = filtered_df.loc[condition.index]  # Select index based on this condition
            loaders[name_of_set] = dataframe_to_loader(df=filtered_df_sn, target=target)

    return loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", required=True, type=str, help="Path to exp parameters")
    return parser.parse_args()


def read_parameters(file: Path, *required) -> dict[str, typing.Any]:
    with open(file) as param:
        params = json.load(param)

    for key in required:
        if key not in params:
            raise ValueError(f"Missing key in parameters : {key}")

    return params


def cross_tab_percent(series_1: pd.Series, series_2: pd.Series):
    """Make a cross tab and display percent for two series"""
    return pd.crosstab(series_1, series_2, margins=True).apply(
        lambda row: row.apply(lambda val: f"{val} ({100 * val / len(series_1):.2f}%)"), axis=1)


def normalize_series(s: pd.Series) -> pd.Series:
    floor_min = math.floor(math.log(1 / 2) / math.log(p.LOW_SMOOTHED_PROB)) + p.EPSILON
    ceil_max = math.ceil(math.log(p.LOW_SMOOTHED_PROB) / math.log(1 / 2)) - p.EPSILON
    return (s - s.min()) / (s.max() - s.min())
    # return np.log((s - s.min()) / (s.max() - s.min()) + p.EPSILON)
    # return np.arctanh(((s-s.min())/(s.max()-s.min())*(1-p.EPSILON - p.EPSILON)+p.EPSILON))
    # return s


def first_key(s: typing.OrderedDict):
    return next(iter(s))


def first_value(s: typing.OrderedDict):
    return s[first_key(s)]


def second_key(s: typing.OrderedDict):
    it = iter(s)
    next(it)
    return next(it)


def second_value(s: typing.OrderedDict):
    return s[second_key(s)]


def get_fairness_criterion_name(method: str) -> str:
    if method == 'demo_par':
        return "Demographic Parity"

    elif method == "eq_opp":
        return "Equality of Opportunity"

    elif method == "avg_odds":
        return "Average Odds"
    else:
        raise ValueError(
            f"Fairness criterion must be either demo_par, eq_opp or avg_odds, got {method}")


# region Data related function : To modify if data get modified

def get_target(db_name: str) -> str:
    """Returns name of the target depending on db used"""
    if db_name == "adult":
        return "income"
    elif db_name == "german":
        return "good_3_bad_2_customer"
    elif db_name == "compas":
        return "two_year_recid"
    elif db_name == "law":
        return "pass_bar"
    elif db_name == "oulad":
        return "final_result"
    elif db_name == "dutch":
        return "occupation"
    elif db_name == "bank":
        return "y"
    elif db_name == "default":
        return "default payment next month"
    elif db_name == "diabetes":
        return "readmitted"
    elif "test" in db_name:  # For testing purpose
        return ""
    else:
        raise ValueError("get_target : Unknown name for database")


def get_map_oc_str(target: str) -> typing.OrderedDict[int, str]:
    """Return str OrderedDict associated with target prediction name. First is favorable class, then unfavorable"""
    map_oc_str = OrderedDict()  # We use OrderedDict to give fav and unfav label in order
    if target == 'income':
        map_oc_str[1] = "hr"
        map_oc_str[0] = "lr"
    elif target == "good_3_bad_2_customer":
        map_oc_str[1] = "good"
        map_oc_str[0] = "bad"
    elif target == "two_year_recid":
        map_oc_str[0] = "nrecid"
        map_oc_str[1] = "recid"
    elif target == "pass_bar":
        map_oc_str[1] = "pass"
        map_oc_str[0] = "fail"
    elif target == "final_result":
        map_oc_str[1] = "pass"
        map_oc_str[0] = "fail"
    elif target == "occupation":
        map_oc_str[1] = "high"
        map_oc_str[0] = "low"
    elif target == "y":
        map_oc_str[1] = "deposit"
        map_oc_str[0] = "no_deposit"
    elif target == "default payment next month":
        map_oc_str[0] = "no_default"
        map_oc_str[1] = "default"
    elif target == "readmitted":  # Don't know what is "favorable" here
        map_oc_str[1] = "less30"
        map_oc_str[0] = "more30"
    else:
        raise ValueError(f'Invalid parameters : got target "{target}"')

    return map_oc_str


def get_map_sc_str(protec_attributes: str) -> typing.OrderedDict[int, str]:
    """Return str map associated with protected attributes name. First is privileged class, then unprivileged"""
    map_sc_str = OrderedDict()  # We use OrderedDict to give priv and unpriv label in order
    if protec_attributes == 'sex':
        map_sc_str[1] = "m"
        map_sc_str[0] = "f"
    elif protec_attributes == 'male':
        map_sc_str[1] = "male"
        map_sc_str[0] = "female"
    elif protec_attributes == 'gender':
        map_sc_str[1] = "m"
        map_sc_str[0] = "f"
    elif protec_attributes == "SEX":
        map_sc_str[1] = "male"
        map_sc_str[0] = "female"
    elif protec_attributes == "Personal_status_and_sex=A92":
        map_sc_str[0] = "male"
        map_sc_str[1] = "female"

    elif protec_attributes == 'race':
        map_sc_str[1] = "white"
        map_sc_str[0] = "nwhite"
    elif protec_attributes == 'race=African-American':
        map_sc_str[0] = "n_afr_amer"
        map_sc_str[1] = "afr_amer"
    elif protec_attributes == 'race=White':
        map_sc_str[1] = "white"
        map_sc_str[0] = "n_white"
    elif protec_attributes == 'racetxt':
        map_sc_str[0] = "white"
        map_sc_str[1] = "nwhite"

    elif protec_attributes == 'age':
        map_sc_str[1] = "old"
        map_sc_str[0] = "young"
    elif protec_attributes == 'age>0.28':
        map_sc_str[1] = "old"
        map_sc_str[0] = "young"
    elif protec_attributes == 'Age_in_years':
        map_sc_str[1] = "old"
        map_sc_str[0] = "young"

    elif protec_attributes == "marital=married":
        map_sc_str[1] = "married"
        map_sc_str[0] = "single"
    elif protec_attributes == "MARRIAGE=2":
        map_sc_str[1] = "single"
        map_sc_str[0] = "other"
    elif protec_attributes == 'education-num':
        map_sc_str[1] = "he"
        map_sc_str[0] = "le"

    else:
        raise ValueError(f'Invalid parameters : got sensitive attribute "{protec_attributes}"')

    return map_sc_str


# endregion

# region Synthetic


def fake_adult_gen(records: int, seed: int = 441999, set_name: str = None) -> pd.DataFrame:
    """
    Generates fake data for dataset Adult https://archive.ics.uci.edu/dataset/2/adult
    @param records: Nombre of records to generate
    @param seed: Seed for replicable random generation
    @param set_name: Choose between 'train' or 'test' for generated inputs. Generate both if None
    @return: pandas Dataframe with random data used in Adult dataset
    """
    citizens = []  # List of results
    # All possible values for each attribute
    work_class = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
                  'Never-worked']
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                 '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
    marital = ['Divorced', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed',
               'Married - AF - spouse']
    relationship = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
    race = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'other']
    occupation = ['Tech-support', ' Craft-repair', ' Other-service', ' Sales', ' Exec-managerial', ' Prof-specialty',
                  ' Handlers-cleaners', ' Machine-op-inspct', ' Adm-clerical', ' Farming-fishing', ' Transport-moving',
                  ' Priv-house-serv', ' Protective-serv', ' Armed-Forces']
    countries = ['United-States', ' Cambodia', ' England', ' Puerto-Rico', ' Canada', ' Germany',
                 ' Outlying-US(Guam-USVI-etc)', ' India', ' Japan', ' Greece', ' South', ' China', ' Cuba', ' Iran',
                 ' Honduras', ' Philippines', ' Italy', ' Poland', ' Jamaica', ' Vietnam', ' Mexico', ' Portugal',
                 ' Ireland', ' France', ' Dominican-Republic', ' Laos', ' Ecuador', ' Taiwan', ' Haiti', ' Columbia',
                 ' Hungary', ' Guatemala', ' Nicaragua', ' Scotland', ' Thailand', ' Yugoslavia', ' El-Salvador',
                 ' Trinadad&Tobago', ' Peru', ' Hong', ' Holand-Netherlands']

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.spawn.html#numpy.random.Generator.spawn
    rng = np.random.default_rng(seed)
    (random_generator_age,
     random_generator_workclass,
     random_generator_fnlwgt,
     random_generator_education,
     random_generator_education_num,
     random_generator_marital,
     random_generator_occupation,
     random_generator_relationship,
     random_generator_race,
     random_generator_sex,
     random_generator_capital_gain,
     random_generator_capital_loss,
     random_generator_hours_per_week,
     random_generator_native_country,
     random_generator_income,
     random_generator_set_name
     ) = rng.spawn(16)

    for i in range(records):
        citizens.append({
            "age": random_generator_age.integers(17, 91),
            "workclass": random_generator_workclass.choice(work_class),
            "fnlwgt": random_generator_fnlwgt.integers(13492, 1490401),
            "education": random_generator_education.choice(education),
            "education-num": random_generator_education_num.integers(1, 17),
            "marital-status": random_generator_marital.choice(marital),
            "occupation": random_generator_occupation.choice(occupation),
            "relationship": random_generator_relationship.choice(relationship),
            "race": random_generator_race.choice(race),
            "sex": random_generator_sex.integers(0, 2),
            "capital-gain": random_generator_capital_gain.integers(0, 12) + random_generator_capital_gain.random(),
            "capital-loss": random_generator_capital_loss.integers(0, 9) + random_generator_capital_loss.random(),
            "hours-per-week": random_generator_hours_per_week.integers(1, 100),
            "native-country": random_generator_native_country.choice(countries),
            "income": random_generator_income.integers(0, 2),
            # Initial proportion for training is 60% training, 20% val et test
            "SetName": set_name or random_generator_set_name.choice(['test', 'train', 'train', 'train', 'valid'])
        })

    return pd.DataFrame(citizens)


def adult_gen_unbias_balanced(records: int, seed: int) -> pd.DataFrame:
    assert records % 4 == 0
    citizens = []  # List of results

    for set_name in ['train', 'test', 'valid']:
        for i in range(int(records / 4)):
            citizens.append({
                "sex": 0,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
            citizens.append({
                "sex": 0,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })

        for i in range(int(records / 4)):
            citizens.append({
                "sex": 1,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
            citizens.append({
                "sex": 1,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })

    random.seed(seed)
    random.shuffle(citizens)

    return pd.DataFrame(citizens)


def adult_gen_unbias_unbalanced(records: int, seed: int) -> pd.DataFrame:
    assert records % 4 == 0
    citizens = []  # List of results

    # 75 % Male
    # 25 % Female
    # 50 % LR / 50% HR

    for set_name in ['train', 'test', 'valid']:
        # F LR
        for i in range(int(records / 8)):
            citizens.append({
                "sex": 0,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
        # F HR
        for i in range(int(records / 8)):
            citizens.append({
                "sex": 0,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })
        # M LR
        for i in range(int(3 * records / 8)):
            citizens.append({
                "sex": 1,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
        # M HR
        for i in range(int(3 * records / 8)):
            citizens.append({
                "sex": 1,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })

    random.seed(seed)
    random.shuffle(citizens)

    return pd.DataFrame(citizens)


def adult_gen_bias_balanced(records: int, seed: int) -> pd.DataFrame:
    assert records % 8 == 0
    citizens = []  # List of results

    for set_name in ['train', 'valid', 'test']:
        for i in range(int(records / 4)):
            citizens.append({
                "sex": 1,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })
        for i in range(int(records / 4)):
            citizens.append({
                "sex": 1,
                "education": 0,
                "income": 1,
                "SetName": set_name
            })
        for i in range(int(records / 4)):
            citizens.append({
                "sex": 0,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
        for i in range(int(records / 4)):
            citizens.append({
                "sex": 0,
                "education": 1,
                "income": 0,
                "SetName": set_name
            })

    random.seed(seed)
    random.shuffle(citizens)

    return pd.DataFrame(citizens)


def adult_gen_bias_unbalanced(seed: int) -> pd.DataFrame:
    citizens = []  # List of results
    prop = {'f_le_lr': 5000,
            'f_le_hr': 0,
            'f_he_lr': 2500,
            'f_he_hr': 2500,
            'm_le_lr': 10000,
            'm_le_hr': 10000,
            'm_he_lr': 0,
            'm_he_hr': 10000}

    for set_name in ['train', 'valid', 'test']:
        for i in range(prop['f_le_lr']):
            citizens.append({
                "sex": 0,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
        for i in range(prop['f_le_hr']):
            citizens.append({
                "sex": 0,
                "education": 0,
                "income": 1,
                "SetName": set_name
            })
        for i in range(prop['f_he_lr']):
            citizens.append({
                "sex": 0,
                "education": 1,
                "income": 0,
                "SetName": set_name
            })
        for i in range(prop['f_he_hr']):
            citizens.append({
                "sex": 0,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })
        for i in range(prop['m_le_lr']):
            citizens.append({
                "sex": 1,
                "education": 0,
                "income": 0,
                "SetName": set_name
            })
        for i in range(prop['m_le_hr']):
            citizens.append({
                "sex": 1,
                "education": 0,
                "income": 1,
                "SetName": set_name
            })
        for i in range(prop['m_he_lr']):
            citizens.append({
                "sex": 1,
                "education": 1,
                "income": 0,
                "SetName": set_name
            })
        for i in range(prop['m_he_hr']):
            citizens.append({
                "sex": 1,
                "education": 1,
                "income": 1,
                "SetName": set_name
            })

    random.seed(seed)
    random.shuffle(citizens)

    return pd.DataFrame(citizens)


def generic_synthetic_data(gen_mode: str, save_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if gen_mode == 'unbias_balanced':
        data = adult_gen_unbias_balanced(records=20000, seed=987654321)
    elif gen_mode == 'unbias_unbalanced':
        data = adult_gen_unbias_unbalanced(records=20000, seed=987654321)
    elif gen_mode == 'bias_balanced':
        data = adult_gen_bias_balanced(records=20000, seed=987654321)
    elif gen_mode == 'bias_unbalanced':
        data = adult_gen_bias_unbalanced(seed=987654321)
    else:
        raise ValueError(f"gen_mode is not recognized, got {gen_mode}")

    # Preprocess
    set_name = data.pop('SetName').squeeze()
    data = data.astype(float)

    # Saving
    data.to_csv(save_path / p.data_filename, index=True, index_label='inputId')
    set_name.to_csv(save_path / p.set_name_filename, index=True, index_label='inputId')

    return data, set_name

# endregion
