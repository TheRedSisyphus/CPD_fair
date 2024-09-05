import os
import warnings

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

import config.parameters as p
from config.logger import create_logger
from src.utils import ATTR_INFO

logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)


def correlation_remover(data: pd.DataFrame, target: str, protec_attr: ATTR_INFO) -> pd.DataFrame:
    # Import made here to speed up ops
    from fairlearn.preprocessing import CorrelationRemover

    attr_name, _, _ = protec_attr

    target_col = data[target]
    cr = CorrelationRemover(sensitive_feature_ids=[attr_name])
    X_cr = cr.fit_transform(data)
    cr_col = list(data.columns)
    cr_col.remove(attr_name)
    X_cr = pd.DataFrame(X_cr, columns=cr_col)
    X_cr[attr_name] = data[attr_name]
    X_cr[target] = target_col
    X_cr = X_cr[list(data.columns)]
    return X_cr


def disparate_impact_remover(data: pd.DataFrame, target: str, protec_attr: ATTR_INFO) -> pd.DataFrame:
    # Import made here to speed up ops
    from aif360.datasets import StandardDataset
    with warnings.catch_warnings(action="ignore"):
        from aif360.algorithms.preprocessing import DisparateImpactRemover

    logger.warning("Before using DIR, make sure that favorable target is 1.0 and that 1.0 is the favored group")

    attr_name, _, _ = protec_attr

    standard_data = StandardDataset(df=data,
                                    label_name=target,
                                    protected_attribute_names=[attr_name],
                                    favorable_classes=[1.0],  # Label that is considered as positive
                                    privileged_classes=[[1.0]])  # protected attr that are considered privileged

    dir_ = DisparateImpactRemover(sensitive_attribute=attr_name)
    data_dir = dir_.fit_transform(standard_data)
    data_dir, _ = data_dir.convert_to_dataframe()
    data_dir = data_dir[list(data.columns)]
    return data_dir


def balance_data(data: pd.DataFrame,
                 protec_attr: ATTR_INFO,
                 mode: str = 'downsampling',
                 seed=987654321) -> pd.DataFrame:
    """Only works for binarized attributes !"""

    # unpacking
    attr_name, attr_1, attr_2 = protec_attr
    value_1: float = attr_1.keys()[0]
    value_2: float = attr_2.keys()[0]
    series_1: pd.Series = attr_1[value_1]
    series_2: pd.Series = attr_2[value_2]

    if mode == 'downsampling':
        min_series: pd.Series = min(series_1, series_2, key=len)
        min_len = len(min_series)
        min_value = min_series.iloc[0]  # value de l'attribut le moins courant

        # new dataset contains all entries for this less frequent attributes
        new_data = data[data[attr_name] == min_value]
        logger.info(f"Balancing data by reducing attribute value to {min_len} inputs")
        # For the other value attribute, we select a random portion of size new data
        portion = data[data[attr_name] != min_value].sample(n=min_len, random_state=seed)  # random portion
        new_data = pd.concat([new_data, portion], ignore_index=True)  # We add this portion to the result

        new_data = new_data.sample(frac=1, random_state=seed)  # Shuffle the dataset
        return new_data


def make_categorical(data, lb: int, ub: int) -> pd.DataFrame:
    """
    :param data:
    :param lb: Below lb values, column are not set categorical
    :param ub: Above ub values, column are not set categorical
    :return: get_dummies is applied on categorical data with number of possible values between lb and ub
    """
    # We select all columns that are not number or bool dtype
    categorical_columns = data.select_dtypes(exclude=['number', 'bool']).columns.tolist()

    # Categorical columns are also numerical columns with 3 to numeric_as_categorical_max_thr different values
    for c in data.select_dtypes(include=['number']).columns:
        if lb <= data[c].value_counts().shape[0] <= ub:
            categorical_columns.append(c)

    logger.info(f"Categorical columns are : {categorical_columns}")
    data = pd.get_dummies(data, columns=categorical_columns, prefix_sep='=')
    return data


def preprocess(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    data.drop(columns=['SetName', 'inputId'], inplace=True, errors='ignore')
    data = make_categorical(data, lb=3, ub=5)
    data = scale_data(data)
    data = data.astype(float)

    # Column order
    target = data.pop(target_name)
    with warnings.catch_warnings(action="ignore"):
        data.insert(len(data.columns), target_name, target)
    return data


def scale_data(data) -> pd.DataFrame:
    scaler = MinMaxScaler()
    numeric_columns = [c for c in data.columns if (is_numeric_dtype(data[c]))]
    if numeric_columns:
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    else:
        logger.warning(f'scale_data - Scaler not applied')
    return data


def generate_db(work_db: str | None,
                save_path: str,
                target: str,
                treatment_param: dict[str, str] | None,
                protec_attr: ATTR_INFO,
                train: bool) -> None:
    """
    :param work_db: Path to the database to work on
    :param save_path:
    :param target :
    :param treatment_param: Python dict, read from JSON file, indicating which treatment to apply
    :param protec_attr: Index of protec attr value
    :param train: True for training db, false for testing db
    :return: Save a new csv at save_path location
    """
    data = pd.read_csv(work_db)
    if train:  # We save set_name before model training
        if 'SetName' not in data:
            set_prop = 0.2
            data_length = len(data)
            data['SetName'] = pd.Series(
                ['valid'] * int(set_prop * data_length) + ['test'] * int(set_prop * data_length) + ['train'] * int(
                    data_length - 2 * data_length * set_prop))

        set_name = data.pop('SetName')
        save_path_sn = os.path.join(os.path.dirname(save_path), "set_name.csv")
        set_name.to_csv(save_path_sn, index=True, index_label="inputId")

    data = preprocess(data=data, target_name=target)
    if treatment_param is not None:
        treatment = treatment_param.get("treatment")
        if treatment == "DIR":
            data = disparate_impact_remover(data=data, protec_attr=protec_attr, target=target)
        elif treatment == "CR":
            data = correlation_remover(data=data, protec_attr=protec_attr, target=target)
        elif treatment == "downsampling":
            data = balance_data(data, protec_attr, mode="downsampling")
        elif treatment == "upsampling":
            raise NotImplementedError

        protec_attr_remove = treatment_param.get("remove_protec_attr")
        if protec_attr_remove:  # If the model is not aware of sensitive groups
            protec_attr_column = data.pop(protec_attr)
        else:
            attr_name, _, _ = protec_attr
            protec_attr_column = data[attr_name]

        if train:  # We save protected attr before model training
            save_path_pa = os.path.join(os.path.dirname(save_path), "protec_attr_index.csv")
            protec_attr_column.astype(int).to_csv(save_path_pa, index=True, index_label="inputId")

    data.to_csv(save_path, index=True, index_label="inputId")
