import os
from pathlib import Path

import pandas as pd

import config.parameters as p
from config.logger import create_logger
from src.operations.get_histograms import get_histograms
from src.operations.likelihood import compute_single_class_likelihood
from src.operations.predictor import Predictor, load_model
from src.utils import parse_args, get_map_sc_str, get_map_oc_str, read_parameters, get_target, get_data_loaders, \
    read_conditional_pa


def get_file_name(filetype: str,
                  dict_sc_str: dict[int, str],
                  dict_oc_str: dict[int, str],
                  input_sc: str | None = None,
                  input_oc: str | None = None,
                  hist_sc: str | None = None,
                  hist_oc: str | None = None,
                  ) -> str:
    """Get the file name """
    filename = filetype
    if input_sc is not None:
        filename += f"_{dict_sc_str[int(input_sc)]}"
    if input_oc is not None:
        filename += f"_{dict_oc_str[int(input_oc)]}"
    if (hist_sc is not None) or (hist_oc is not None):
        filename += '-'
        if (hist_sc is not None) and (hist_oc is not None):
            filename += f"{dict_sc_str[int(hist_sc)]}_{dict_oc_str[int(hist_oc)]}"
        elif hist_sc is None:
            filename += f"{dict_oc_str[int(hist_oc)]}"
        elif hist_oc is None:
            filename += f"{dict_sc_str[int(hist_sc)]}"

    return filename + ".csv"


if __name__ == "__main__":
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)

    params_data = read_parameters(experiment_dir.parent / p.data_dir_name / 'parameters.json')

    params = read_parameters(args.param, "extraction_layer",
                             "contrib_oc_h1", "contrib_sc_h1",
                             "contrib_oc_h2", "contrib_sc_h2")

    target = get_target(params_data["db_name"]) if params_data.get("db_name") else params_data["target"]

    # endregion

    # region 0. Data Loaders and predictor

    df = pd.read_csv(experiment_dir.parent / p.data_dir_name / p.data_filename, index_col='inputId')
    set_name_column = pd.read_csv(experiment_dir.parent / p.data_dir_name / p.set_name_filename, index_col='inputId')
    set_name_column = set_name_column.squeeze()
    loaders = get_data_loaders(df=df, set_name=set_name_column, target=target)

    pred = Predictor(dimensions=load_model(experiment_dir.parent / p.data_dir_name / p.model_path))

    # endregion

    map_oc_str = get_map_oc_str(target)
    attr_name, _, _ = read_conditional_pa(name_pa=params_data['pa_name'], data=df)
    map_sc_str = get_map_sc_str(attr_name)

    # region Not random

    contrib_path_1 = get_file_name(filetype="contribs",
                                   dict_sc_str=map_sc_str,
                                   dict_oc_str=map_oc_str,
                                   input_sc=params["contrib_sc_h1"],
                                   input_oc=params["contrib_oc_h1"])

    pred.save_activation_levels(data_loader=loaders['train'],
                                index_path=experiment_dir.parent / p.data_dir_name / p.indexes_filename,
                                save_path=experiment_dir / contrib_path_1,
                                layer_id=params["extraction_layer"],
                                set_name='train',
                                correct=True,
                                output_class=params["contrib_oc_h1"],
                                sensitive_class=params["contrib_sc_h1"])

    contrib_path_2 = get_file_name(filetype="contribs",
                                   dict_sc_str=map_sc_str,
                                   dict_oc_str=map_oc_str,
                                   input_sc=params["contrib_sc_h2"],
                                   input_oc=params["contrib_oc_h2"])

    pred.save_activation_levels(data_loader=loaders['train'],
                                index_path=experiment_dir.parent / p.data_dir_name / p.indexes_filename,
                                save_path=experiment_dir / contrib_path_2,
                                layer_id=params["extraction_layer"],
                                set_name='train',
                                correct=True,
                                output_class=params["contrib_oc_h2"],
                                sensitive_class=params["contrib_sc_h2"])

    # endregion

    # region random

    pred.save_activation_levels(data_loader=loaders['train'],
                                index_path=experiment_dir.parent / p.data_dir_name / p.indexes_filename,
                                save_path=experiment_dir / "name",
                                layer_id=params["extraction_layer"],
                                set_name='train',
                                correct=True,
                                rand=True)

    # endregion
