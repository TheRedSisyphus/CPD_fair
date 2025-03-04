from pathlib import Path

import pandas as pd
from tabulate import tabulate

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor
from src.operations.preprocess import Preprocess, ProtectedAttr
from src.readers import file_reader
from src.utils import parse_args, cross_tab_percent, get_map_oc_str, get_target, get_map_sc_str, read_parameters, \
    get_data_loaders, filter_indexes


def log_data_info(df: pd.DataFrame, set_name: pd.Series, pa_series: pd.Series, target: str, pa_name: str) -> None:
    """
    To log various information on data, such as the proportions of target and protected attribute
    in training, testing and valid set
    """
    # Getting str representation of outcome classes and protected groups
    oc_dict_mapping = get_map_oc_str(target)
    sc_dict_mapping = get_map_sc_str(pa_name)

    r, c = df.shape
    logger.info(
        f"{r} inputs, {c} features after preprocessing, including task of prediction {target}.")

    # Log info for All data
    target_col = df[target].astype(int).map(oc_dict_mapping)
    protected_attr = pa_series.astype(int).map(sc_dict_mapping)
    logger.info(
        f"Proportions : \n\n{tabulate(cross_tab_percent(target_col, protected_attr), headers="keys", tablefmt='github')}\n")

    # Log info for Train data
    train_data = df[set_name == 'train']
    logger.info(f"Training set of {train_data.shape[0]} inputs ({100 * train_data.shape[0] / r:.2f}% of original data)")
    target_train = target_col[set_name == 'train']
    protected_attr_train = protected_attr[set_name == 'train']
    logger.info(
        f"Proportions : \n\n{tabulate(cross_tab_percent(target_train, protected_attr_train), headers="keys", tablefmt='github')}\n")

    # Log info for Valid data
    valid_data = df[set_name == 'valid']
    logger.info(f"Valid set of {valid_data.shape[0]} inputs ({100 * valid_data.shape[0] / r:.2f}% of original data)")
    target_valid = target_col[set_name == 'valid']
    protected_attr_valid = protected_attr[set_name == 'valid']
    logger.info(
        f"Proportions : \n\n{tabulate(cross_tab_percent(target_valid, protected_attr_valid), headers="keys", tablefmt='github')}\n")

    # Log info for Test data
    test_data = df[set_name == 'test']
    logger.info(f"Test set of {test_data.shape[0]} inputs ({100 * test_data.shape[0] / r:.2f}% of original data)")
    target_test = target_col[set_name == 'test']
    protected_attr_test = protected_attr[set_name == 'test']
    logger.info(
        f"Proportions : \n\n{tabulate(cross_tab_percent(target_test, protected_attr_test), headers="keys", tablefmt='github')}\n")


def log_model_info(index_path: Path, target_name: str, pa_name: str) -> None:
    """
    To log various information on the model, such as the precision on training, testing and valid set
    """
    indexes = file_reader(path=index_path, header=p.indexes_header)
    # Getting str representation of outcome classes and protected groups
    oc_dict_mapping = get_map_oc_str(target_name)
    sc_dict_mapping = get_map_sc_str(pa_name)

    # None represents the union of all possible values
    target_values = [None] + list(set([int(line[p.true_class_pos]) for line in indexes]))
    pa_values = [None] + list(set([int(line[p.sens_attr_pos]) for line in indexes]))
    for target_value in target_values:
        for pa_value in pa_values:
            for set_name in [None, 'train', 'valid', 'test']:
                set_str = "" if set_name == 'all' else f" ({set_name} set)"
                target_str = "" if target_value is None else f"{oc_dict_mapping[target_value]} "
                pa_str = "" if pa_value is None else f"{sc_dict_mapping[pa_value]} "
                set_index = [line for line in indexes if
                             line[p.set_name_pos] == set_name] if set_name is not None else indexes
                correct_pred = len([line for line in set_index if line[p.true_class_pos] == line[p.pred_class_pos]])
                logger.info(
                    f"Model precision on {pa_str}{target_str}inputs{set_str}: {(correct_pred / len(set_index)) * 100:.2f}%")


def log_fair_metrics(index_path: Path, favorable_class: int, privileged_class: int) -> None:
    """
    To log various information on the fairness metrics,
    such as the disparate impact, the demographic parity difference and the equalized opportunity difference
    """
    indexes = file_reader(path=index_path, header=p.indexes_header)

    for set_name in [None, 'train', 'valid', 'test']:
        ind_filter = filter_indexes(indexes, set_name=set_name)
        filtered_indexes = [line for line in indexes if int(line[p.input_id_pos]) in ind_filter]
        set_str = "" if set_name is None else f" on {set_name} set"

        # Number of unprivileged profiles
        unpriv = len([line for line in filtered_indexes if int(line[p.sens_attr_pos]) != privileged_class])
        # Number of privileged profiles
        priv = len([line for line in filtered_indexes if int(line[p.sens_attr_pos]) == privileged_class])
        # Number of unprivileged profiles with favorables predictions
        fav_outcomes_unpriv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) != privileged_class)])
        # Number of privileged profiles with favorables predictions
        fav_outcomes_priv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) == privileged_class)])
        # Number of unprivileged profiles with favorables classes and predictions
        fav_true_fav_pred_unpriv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(
            line[p.true_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) == privileged_class)])
        # Number of privileged profiles with favorables classes and predictions
        fav_true_fav_pred_priv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(
            line[p.true_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) != privileged_class)])
        # Number of unprivileged profiles with favorables predictions
        fav_pred_unpriv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) != privileged_class)])
        # Number of privileged profiles with favorables predictions
        fav_pred_priv = len([line for line in filtered_indexes if (
                int(line[p.pred_class_pos]) == favorable_class and int(line[p.sens_attr_pos]) == privileged_class)])

        if unpriv != 0 and priv != 0 and fav_outcomes_priv != 0 and fav_outcomes_unpriv != 0:
            logger.info(f"Disparate Impact{set_str} is {(fav_outcomes_unpriv / unpriv) / (fav_outcomes_priv / priv)}")
            logger.info(
                f"Demographic Parity Difference{set_str} is {(fav_outcomes_unpriv / unpriv) - (fav_outcomes_priv / priv)}")
        else:
            logger.warning("Only one class of sensitive attribute found")
        if fav_pred_unpriv != 0 and fav_pred_priv != 0:
            logger.info(
                f"Equalized Opportunity Difference{set_str} is {(fav_true_fav_pred_unpriv / fav_pred_unpriv) - (fav_true_fav_pred_priv / fav_pred_priv)}")
        else:
            logger.warning("One class of sensitive attribute is never favored by the model")


if __name__ == "__main__":
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)
    params = read_parameters(args.param,
                             'db_name',
                             'sanitization_level',
                             'balance',
                             'pa_name',
                             'cr_attr_name',
                             'dir_attr_name',
                             'favorable_classes',
                             'privileged_classes',
                             'model_dimensions',
                             'model_lr',
                             'model_epochs')

    # region Preprocess

    prepro = Preprocess(db_name=params["db_name"],
                        save_dir=experiment_dir,
                        sanitization_level=params["sanitization_level"],
                        balance=params["balance"],
                        cr_attr_name=params["cr_attr_name"],
                        dir_attr_name=params["dir_attr_name"],
                        favorable_classes=params["favorable_classes"],
                        privileged_classes=params["privileged_classes"])

    # endregion

    # region Train model

    loaders = get_data_loaders(df=prepro.df, set_name=prepro.set_name, target=prepro.target)

    model = Predictor(dimensions=params["model_dimensions"], lr=params["model_lr"],
                      c_weight=prepro.c_weights if params.get("c_weights") else None)

    model.train_model(epochs=params["model_epochs"],
                      save_path=experiment_dir / p.model_path,
                      train_loader=loaders['train'],
                      valid_loader=loaders['valid'])

    # endregion

    # region Save PA

    pa = ProtectedAttr(name=params["pa_name"],
                       data_path=Path(experiment_dir) / p.data_filename,
                       set_name_path=Path(experiment_dir) / p.set_name_filename,
                       save_dir=experiment_dir,
                       target=get_target(params['db_name']))

    attr_name = pa.save_protected_attribute()

    pa.write_indexes(model=model)  # Write and save indexes

    # end region

    # region Log informations

    set_name_column = pd.read_csv(experiment_dir / p.set_name_filename, index_col='inputId')
    set_name_column = set_name_column.squeeze()

    pa_column = pd.read_csv(experiment_dir / p.protec_attr_filename, index_col='inputId')
    pa_column = pa_column.squeeze()

    log_data_info(df=prepro.df, set_name=set_name_column, pa_series=pa_column, target=prepro.target,
                  pa_name=attr_name)

    log_model_info(index_path=experiment_dir / p.indexes_filename,
                   target_name=prepro.target,
                   pa_name=attr_name)

    if params.get("favorable_class") and params.get("privileged_class"):
        log_fair_metrics(index_path=experiment_dir / p.indexes_filename,
                         favorable_class=int(params.get("favorable_class")),
                         privileged_class=int(params.get("privileged_class")))
    else:
        logger.info("Fairness metrics were not computed (specify favorable class and privileged_class)")

    # endregion
