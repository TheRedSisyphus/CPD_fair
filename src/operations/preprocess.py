import csv
import random
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor
from src.utils import get_target, get_data_loaders, read_conditional_pa

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


class Preprocess:
    """Preprocess object that reads the data, and treats it to be used by a torch.Module"""

    def __init__(self, db_name: str, save_dir: Path,
                 sanitization_level: float | None = None,
                 balance: list[dict[str, str | list[str]]] | None = None,
                 cr_attr_name: str | None = None,
                 dir_attr_name: str | None = None,
                 favorable_classes: list | None = None,
                 privileged_classes: list[list] | None = None,
                 test: bool = False):

        if cr_attr_name is not None and dir_attr_name is not None:
            raise ValueError("Preprocess : CR and DIR are not made for concurrent usage")
        if (cr_attr_name is not None or dir_attr_name is not None) and sanitization_level is not None:
            raise ValueError("Preprocess : CR or DIR are not made for concurrent usage with GanSan")
        if not (bool(dir_attr_name) and bool(favorable_classes) and bool(privileged_classes)):
            if bool(dir_attr_name) or bool(favorable_classes) or bool(privileged_classes):
                raise ValueError("Preprocess : Not all arguments has been specified for DIR")

        # Database name, used to read database in the db_refined_dir directory
        self.db_name: str = db_name
        if test:  # For testing purpose
            db_path = (p.test_db_dir / self.db_name).with_suffix('.csv')
        else:
            db_path = (p.db_refined_dir / self.db_name).with_suffix('.csv')
        self.df: pd.DataFrame = pd.read_csv(db_path, index_col='inputId')  # Read refined csv

        # c_charge_desc in compas dataset in the str description of the charge. Not useful for prediction
        if self.db_name == 'compas':  # todo
            self.df.drop(columns='c_charge_desc', inplace=True)

        # Name of the target prediction
        self.target: str = get_target(self.db_name)
        self.save_dir: Path = Path(save_dir)  # Path to dir where to save all output

        self.sanitization_level: float | None = sanitization_level  # If None, no sanitization
        self.balance: list[dict[str, str | list[str]]] | None = balance  # If None, no balancing
        self.set_name: pd.Series | None = None  # Indicate which input is in which set (train, test...)

        class_w = self.df[self.df[self.target] == 0].shape[0]/self.df.shape[0]
        self.c_weights = [class_w, 1 - class_w]  # this the class weights, for unbalanced dataset

        # Data treatment
        if not test:  # Manually test operations during testing
            self.set_setName()  # Indicate which input is in which set (train, test...)
            self.sanitize()  # Sanitizing test set if needed
            self.balance_data()  # Balancing data if needed
            self.preprocess()  # Scale the data and transform categorical column into numerical
            self.correlation_remover(cr_attr_name)  # Apply Correlation Remover
            # Apply Disparate Impact Remover
            self.disparate_impact_remover(protec_attr=dir_attr_name,
                                          favorable_classes=favorable_classes,
                                          privileged_classes=privileged_classes)
            self.save_all()  # Save all data in save_dir

    def set_setName(self) -> None:
        """Read which inputs belong to which 'set' (training set, testing set, valid set).
        If 'SetName' columns is not in dataset, separate data into 60% training, 20% valid and 20% testing set"""
        if 'SetName' not in self.df:
            set_prop = 0.2
            data_length = len(self.df)
            self.df['SetName'] = pd.Series(
                ['valid'] * int(set_prop * data_length) + ['test'] * int(set_prop * data_length) + ['train'] * int(
                    data_length - 2 * data_length * set_prop))
            self.df['SetName'] = self.df['SetName'].fillna('train')  # Sometimes, not all input are given a value

        self.set_name = self.df.pop('SetName')
        self.set_name = self.set_name.squeeze()  # Make a pd.Series rather than pd.DataFrame

    def sanitize(self) -> None:
        """Reads sanitized data and replace testing set from self.df with new data"""
        # Does nothing if sanitization_level is None
        if self.sanitization_level is not None:
            sanitize_path = (p.db_refined_dir / (
                    self.db_name + '_' + str(self.sanitization_level).replace('.', ''))).with_suffix(
                '.csv')
            sanitized_df = pd.read_csv(sanitize_path, index_col='inputId')
            if 'SetName' in sanitized_df.columns:
                set_name_san = sanitized_df.pop('SetName')
                if not set_name_san.equals(self.set_name):
                    raise ValueError("SetName column is different between original and sanitized data.")

            # We only sanitize testing set
            sanitized_df = sanitized_df[self.set_name == 'test']

            # Reindex sanitized data to make the index corresponds to testing set
            sanitized_df = sanitized_df.set_index(
                self.df.loc[self.df.index.isin(sanitized_df.index), :].index)

            # Set data type identical between original data and sanitized data (to avoid warning)
            for c in sanitized_df.columns:
                if c in self.df.columns:
                    if sanitized_df[c].dtype != self.df[c].dtype:
                        try:
                            logger.info(
                                f"Parsing column {c} from type {self.df[c].dtype} to type {sanitized_df[c].dtype}")
                            self.df[c] = self.df[c].astype(sanitized_df[c].dtype)
                        except ValueError:
                            logger.warning(
                                f"Impossible to parse column {c} from type {self.df[c].dtype} to type {sanitized_df[c].dtype}. "
                                "Please check data before continuing")
                elif c == 'c_charge_desc':  # Todo
                    sanitized_df.drop(columns='c_charge_desc', inplace=True)
                else:
                    raise ValueError(
                        f"ERROR sanitize : sanitize data doesn't have the same column {c} as original data")

            # Replace non-sanitized testing set with sanitized testing set
            self.df.update(sanitized_df)
            logger.info(f"Sanitize data with sanitization level {self.sanitization_level} ({sanitize_path.stem})")

    def balance_data(self, seed=987654321) -> None:
        """Balance dataset along an attribute. Either Downsampling or upsampling.
        @:param seed: Seed for reproducibility of random operations"""
        # Does nothing if balance is None
        if self.balance is not None and len(self.balance) > 0:
            for balancing in self.balance:
                for attribute, balance_mode in balancing.items():
                    if balance_mode in ["d", "down", "downsampling"]:
                        if isinstance(attribute, str):  # Independent balancing
                            frames = []
                            logger.info(f"BALANCE_DATA : Downsampling to balance {attribute} attributes")
                            if attribute not in self.df:
                                raise ValueError(f"Attribute {attribute} not in dataframe")
                            value_counts = self.df[attribute].value_counts()
                            less_frequent = value_counts.index.to_list()[-1]
                            filtered_df = self.df[self.df[attribute] == less_frequent]
                            target_len = len(filtered_df)
                            frames.append(filtered_df)
                            logger.info(
                                f"BALANCE_DATA : Less frequent value is {less_frequent} in {target_len} copies")

                            for other_value in value_counts.index.to_list()[:-1]:
                                # random portion
                                portion = self.df[self.df[attribute] == other_value].sample(n=target_len,
                                                                                            random_state=seed)
                                frames.append(portion)

                            self.df = pd.concat(frames).sort_index()  # Index is the same order as original dataset

                            # Update set_name
                            self.set_name = self.set_name.loc[self.df.index]

                        elif isinstance(attribute, list):  # Todo : intersectional balancing
                            raise NotImplementedError

                    elif balance_mode in ["u", "up", "upsampling"]:
                        if isinstance(attribute, str):  # Independent balancing
                            frames = []
                            portions = []
                            if attribute not in self.df:
                                raise ValueError(f"Attribute {attribute} not in dataframe")
                            logger.info(f"BALANCE_DATA : Upsampling to balance {attribute} attributes")
                            value_counts = self.df[attribute].value_counts()
                            most_frequent = value_counts.index.to_list()[0]
                            filtered_df = self.df[self.df[attribute] == most_frequent]
                            target_len = len(filtered_df)
                            frames.append(filtered_df)
                            logger.info(
                                f"BALANCE_DATA : Most frequent value is {most_frequent} in {target_len} copies")

                            for other_value in value_counts.index.to_list()[1:]:
                                less_freq_df = self.df[self.df[attribute] == other_value]
                                len_less_freq = len(less_freq_df)
                                frames.append(less_freq_df)
                                len_to_add = target_len - len_less_freq
                                if len_to_add < 0:
                                    raise ValueError(
                                        f"Less frequent attribute {attribute}={other_value} is more frequent than {attribute}={most_frequent}")
                                elif len_to_add == 0:
                                    pass
                                elif len_to_add <= len_less_freq:
                                    portion = less_freq_df.sample(n=len_to_add, random_state=seed)
                                    portions.append(portion)

                                else:  # On doit ajouter plus d'une fois l'ensemble des éléments moins courant
                                    repetition = len_to_add // len_less_freq
                                    remainder = len_to_add % len_less_freq
                                    for r in range(repetition):
                                        portions.append(less_freq_df)
                                    portion = less_freq_df.sample(n=remainder, random_state=seed)
                                    portions.append(portion)

                            frames_df = pd.concat(frames)
                            if len(portions) > 0:
                                portions = pd.concat(portions, ignore_index=True)
                                # Upsampled elements are added at the end with the highest indexes
                                portions.index += len(self.df)
                                self.df = pd.concat([frames_df, portions]).sort_index()

                                self.df = self.df.sort_index()

                                # Update set_name_path
                                # Todo : Avoid code repetition
                                len_portions = len(portions)
                                set_prop = 0.2
                                new_set_name = ['valid'] * int(set_prop * len_portions) + ['test'] * int(
                                    set_prop * len_portions) + ['train'] * int(
                                    len_portions - 2 * len_portions * set_prop)
                                if len(new_set_name) < len_portions:
                                    new_set_name += ['train'] * int(len_portions - len(new_set_name))
                                random.shuffle(new_set_name)
                                set_name_added = pd.Series(data=new_set_name,
                                                           index=portions.index)
                                self.set_name = pd.concat([self.set_name, set_name_added])
                            self.set_name.index.name = 'inputId'
                        elif isinstance(attribute, list):  # Todo : intersectional balancing
                            raise NotImplementedError

                    else:
                        raise ValueError(f"Preprocess : Invalid arg balance, got {balance_mode}")

    # region Preprocess

    def make_categorical(self, lb: int, ub: int) -> None:
        """
        :param lb: Below lb values, column are not set categorical
        :param ub: Above ub values, column are not set categorical
        :return: get_dummies is applied on categorical data with number of possible values between lb and ub
        """
        # We select all columns that are not number or bool dtype
        categorical_columns = self.df.select_dtypes(exclude=['number', 'bool']).columns.tolist()

        # Categorical columns are also numerical columns with 3 to numeric_as_categorical_max_thr different values
        for c in self.df.select_dtypes(include=['number']).columns:
            if lb <= self.df[c].value_counts().shape[0] <= ub:
                categorical_columns.append(c)

        logger.info(f"Categorical columns are : {categorical_columns}")
        self.df = pd.get_dummies(self.df, columns=categorical_columns, prefix_sep='=')

    def scale_data(self) -> None:
        """Scale all numerical column between 0 and 1"""
        scaler = MinMaxScaler()
        numeric_columns = [c for c in self.df.columns if (is_numeric_dtype(self.df[c]))]
        if numeric_columns:
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        else:
            logger.warning(f'SCALE_DATA : scale_data - Scaler not applied')

    def preprocess(self) -> None:
        """Calls sequentially : make_categorical and scale_data.
        Parse to float the data and move target to last column"""
        if self.db_name == 'default':
            upper_b = 11
        elif self.db_name == 'dutch':
            upper_b = 6
        elif self.db_name == 'oulad':
            upper_b = 7
        else:
            upper_b = 5
        self.make_categorical(lb=3, ub=upper_b)

        self.scale_data()
        self.df = self.df.astype(float)

        # Column order : put target at the end
        target = self.df.pop(self.target)
        with warnings.catch_warnings(action="ignore"):
            self.df.insert(len(self.df.columns), self.target, target)

    def save_all(self) -> None:
        """Save current pandas Dataframe and information about setName"""
        self.df.to_csv(self.save_dir / p.data_filename, index=True, index_label='inputId')
        self.set_name.to_csv(self.save_dir / p.set_name_filename)

    # endregion

    def remove_attributes(self, remove_attr: list[str] | str) -> None:
        # Pour le moment on ne veut pas de ça
        # if isinstance(remove_attr, str):
        #     remove_attr = [remove_attr]
        # for attr in remove_attr:
        #     self.df.pop(attr)
        #     logger.info(f"Attribute {remove_attr} has been removed from data")
        raise NotImplementedError

    def correlation_remover(self, protec_attr: str | None) -> None:
        """Apply correlation remover algorithm for Fairlearn to the data"""
        if protec_attr is not None:
            from fairlearn.preprocessing import CorrelationRemover

            target_col = self.df[self.target]
            index_col = self.df.index

            cr = CorrelationRemover(sensitive_feature_ids=[protec_attr])
            X_cr = cr.fit_transform(X=self.df, y=target_col)
            cr_col = list(self.df.columns)
            cr_col.remove(protec_attr)
            X_cr = pd.DataFrame(X_cr, columns=cr_col)
            X_cr.index = index_col
            X_cr[protec_attr] = self.df[protec_attr]
            X_cr[self.target] = target_col

            # column order
            X_cr = X_cr[list(self.df.columns)]
            self.df = X_cr
            logger.info(f"Use Correlation Remover on {protec_attr}")

    def disparate_impact_remover(self,
                                 protec_attr: str | None,
                                 favorable_classes: list | None,
                                 privileged_classes: list[list] | None) -> None:
        """Apply disparate impact remover algorithm for AIF360 to the data"""

        if protec_attr is not None and favorable_classes is not None and privileged_classes is not None:
            from aif360.datasets import StandardDataset
            with warnings.catch_warnings(action="ignore"):
                from aif360.algorithms.preprocessing import DisparateImpactRemover

            standard_data = StandardDataset(df=self.df,
                                            label_name=self.target,
                                            protected_attribute_names=[protec_attr],
                                            favorable_classes=favorable_classes,  # Label that is considered as positive
                                            privileged_classes=privileged_classes)  # protected attr that are considered privileged

            dir_ = DisparateImpactRemover(sensitive_attribute=protec_attr)
            data_dir = dir_.fit_transform(standard_data)
            data_dir, _ = data_dir.convert_to_dataframe()
            data_dir = data_dir[list(self.df.columns)]
            self.df = data_dir
            self.df.index = self.df.index.astype(int)
            logger.info(f"Use Disparate Impact Remover on {protec_attr}")


class ProtectedAttr:
    """Class that handle which protected attribute is studied"""

    def __init__(self, name: str, target: str, data_path: Path, set_name_path: Path, save_dir: Path):
        self.df: pd.DataFrame = pd.read_csv(data_path, index_col='inputId')
        self.target: str = target  # Name of target for prediction
        # Information about 'set' (training, testing, valid)
        self.set_name = pd.read_csv(set_name_path, index_col='inputId')
        self.set_name: pd.Series = self.set_name.squeeze()
        self.save_dir: Path = save_dir
        # Name of the protected attributes
        self.name: str = name
        # Series that link ID to protected groups
        self.study_group: pd.Series | None = None

    def save_protected_attribute(self) -> str:
        """Create and save in save_dir the protected group for each ID.
        Read basic condition if protected feature is numerical and not categorical"""
        if self.name in self.df.columns:
            self.study_group = self.df[self.name].squeeze()
        else:
            attr_name, _, self.study_group = read_conditional_pa(name_pa=self.name, data=self.df)

        save_path_pa = self.save_dir / p.protec_attr_filename
        self.study_group.astype(int).to_csv(save_path_pa, index=True, index_label="inputId")
        logger.info(f"Saved Protected Attribute at {save_path_pa}")

        return self.name if self.name in self.df.columns else attr_name

    def write_index_set(self, model: Predictor, name_of_set: str) -> list[list[Any]]:
        """Write the indexes for a given 'set' (training, testing, valid)"""
        content = []
        # Filter on set Name
        data = self.df[self.set_name == name_of_set]

        loaders = get_data_loaders(df=self.df, set_name=self.set_name, target=self.target)

        _, is_pred_correct = model.test(loaders[name_of_set])

        row_count = 0  # Row counter, may be different from index
        for index, row in data.iterrows():
            predicted_class = not bool(is_pred_correct[row_count]) ^ bool(row[self.target])
            content.append(
                [index, int(row[self.target]), int(predicted_class), int(self.study_group.loc[index]), name_of_set])
            row_count += 1

        return content

    def write_indexes(self, model: Predictor) -> None:
        """Save in save dir the 'indexes', csv file containing all important info for each ID :
        ID, the ground truth class, class predicted by model, the protected group and the 'set' (training, testing or valid)"""
        with open(self.save_dir / p.indexes_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['inputId', 'TrueClass', 'PredictedClass', 'SensitiveAttr', 'SetName'])

            for set_name in ["valid", "test", "train"]:
                logger.info(f"WRITE_INDEXES : write index for {set_name} set")
                content = self.write_index_set(model=model, name_of_set=set_name)
                for row in content:
                    writer.writerow(row)
