import json
import os
from typing import Any

import config.parameters as p
from config.logger import create_logger
from src.operations.get_histograms import get_histograms
from src.operations.likelihood import compute_single_class_likelihood
from src.operations.predictor import Predictor, load_model, get_data_loader
from src.utils import parse_args, get_map_str, get_target, get_protec_attr


def get_file_name(filetype: str,
                  dict_sc_str: dict[str, str],
                  dict_oc_str: dict[str, str],
                  input_sc: str | None = None,
                  input_oc: str | None = None,
                  hist_sc: str | None = None,
                  hist_oc: str | None = None,
                  ) -> str:
    filename = filetype
    if input_sc is not None:
        filename += f"_{dict_sc_str[input_sc]}"
    if input_oc is not None:
        filename += f"_{dict_oc_str[input_oc]}"
    if (hist_sc is not None) or (hist_oc is not None):
        filename += '-'
        if (hist_sc is not None) and (hist_oc is not None):
            filename += f"{dict_sc_str[hist_sc]}_{dict_oc_str[hist_oc]}"
        elif hist_sc is None:
            filename += f"{dict_oc_str[hist_oc]}"
        elif hist_oc is None:
            filename += f"{dict_sc_str[hist_sc]}"

    return filename + ".txt"


def read_parameters(file: str) -> dict[str, Any]:
    """Read parameters file and return python dict with absolute path using config file"""
    with open(file) as param:
        param_dict = json.load(param)

    for key in ['extraction_layer',
                'contrib_oc_h1', 'contrib_sc_h1',
                'contrib_oc_h2', 'contrib_sc_h2',
                "contrib_set_name_lh",
                "contrib_correct_lh",
                "contrib_oc_lh",
                "contrib_sc_lh"]:
        if key not in param_dict:
            raise ValueError(f"Missing key in parameters : {key}")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(file)), "_data")
    data_param_file = os.path.join(data_dir, 'parameters.json')
    with open(data_param_file) as data_param:
        data_param_dict = json.load(data_param)

    db_name = data_param_dict['work_db']
    param_dict['target'] = get_target(db_name)
    attr_name = get_protec_attr(data_param_dict['protec_attr'])
    param_dict['sens_attr'] = attr_name

    param_dict["train_db_path"] = os.path.join(data_dir, "train_data.csv")  # Todo: put in config file filename
    param_dict["test_db_path"] = os.path.join(data_dir, "test_data.csv")
    param_dict["sn_path"] = os.path.join(data_dir, "set_name.csv")

    param_dict["model_path"] = os.path.join(data_dir, db_name + ".pt")
    param_dict["index_path"] = os.path.join(data_dir, "indexes.txt")

    return param_dict


if __name__ == "__main__":
    args = parse_args()
    experiment_dir = os.path.dirname(args.param)
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL, file_dir=experiment_dir)

    # region 0. PARAMETERS
    params = read_parameters(args.param)

    map_sc_str, map_oc_str = get_map_str(params)

    # endregion

    # region 0. Data Loaders and predictor

    loaders = get_data_loader(train_data_path=params["train_db_path"],
                              test_data_path=params["test_db_path"],
                              set_name_path=params["sn_path"],
                              target=params["target"])
    pred = Predictor(dimensions=load_model(params["model_path"]))

    # endregion

    # region 1. First histogram
    contrib_path_1 = get_file_name(filetype="contribs",
                                   dict_sc_str=map_sc_str,
                                   dict_oc_str=map_oc_str,
                                   input_sc=params["contrib_sc_h1"],
                                   input_oc=params["contrib_oc_h1"])

    pred.save_activation_levels(data_loader=loaders['train_db']['all'],
                                index_path=params["index_path"],
                                save_path=str(os.path.join(experiment_dir, contrib_path_1)),
                                layer_id=params["extraction_layer"],
                                set_name='train',
                                correct=True,
                                output_class=params["contrib_oc_h1"],
                                sensitive_class=params["contrib_sc_h1"])

    hist_path_1 = get_file_name(filetype="hist",
                                dict_sc_str=map_sc_str,
                                dict_oc_str=map_oc_str,
                                input_sc=params["contrib_sc_h1"],
                                input_oc=params["contrib_oc_h1"])
    get_histograms(contribs_path=str(os.path.join(experiment_dir, contrib_path_1)),
                   model_structure=pred.structure[1],
                   save_path=str(os.path.join(experiment_dir, hist_path_1)))
    # endregion

    # region 2. Second histogram
    contrib_path_2 = get_file_name(filetype="contribs",
                                   dict_sc_str=map_sc_str,
                                   dict_oc_str=map_oc_str,
                                   input_sc=params["contrib_sc_h2"],
                                   input_oc=params["contrib_oc_h2"])
    pred.save_activation_levels(data_loader=loaders['train_db']['all'],
                                index_path=params["index_path"],
                                save_path=str(os.path.join(experiment_dir, contrib_path_2)),
                                layer_id=params["extraction_layer"],
                                set_name='train',
                                correct=True,
                                output_class=params["contrib_oc_h2"],
                                sensitive_class=params["contrib_sc_h2"])

    hist_path_2 = get_file_name(filetype="hist",
                                dict_sc_str=map_sc_str,
                                dict_oc_str=map_oc_str,
                                input_sc=params["contrib_sc_h2"],
                                input_oc=params["contrib_oc_h2"])
    get_histograms(contribs_path=str(os.path.join(experiment_dir, contrib_path_2)),
                   model_structure=pred.structure[1],
                   save_path=str(os.path.join(experiment_dir, hist_path_2)))
    # endregion

    # region 2. Extract CPL
    contrib_path_lh = get_file_name(filetype="contribs",
                                    dict_sc_str=map_sc_str,
                                    dict_oc_str=map_oc_str,
                                    input_sc=params["contrib_sc_lh"],
                                    input_oc=params["contrib_oc_lh"])

    pred.save_activation_levels(data_loader=loaders['test_db']['all'],
                                index_path=params["index_path"],
                                save_path=str(os.path.join(experiment_dir, contrib_path_lh)),
                                layer_id=params["extraction_layer"],
                                set_name=params["contrib_set_name_lh"],
                                correct=params["contrib_correct_lh"],
                                output_class=params["contrib_oc_lh"],
                                sensitive_class=params["contrib_sc_lh"])

    lh_path_1 = get_file_name(filetype="lh",
                              dict_sc_str=map_sc_str,
                              dict_oc_str=map_oc_str,
                              input_sc=params["contrib_sc_lh"],
                              input_oc=params["contrib_oc_lh"],
                              hist_sc=params["contrib_sc_h1"],
                              hist_oc=params["contrib_oc_h1"])
    compute_single_class_likelihood(contribs_path=str(os.path.join(experiment_dir, contrib_path_lh)),
                                    hist_path=str(os.path.join(experiment_dir, hist_path_1)),
                                    save_path=str(os.path.join(experiment_dir, lh_path_1)))

    lh_path_2 = get_file_name(filetype="lh",
                              dict_sc_str=map_sc_str,
                              dict_oc_str=map_oc_str,
                              input_sc=params["contrib_sc_lh"],
                              input_oc=params["contrib_oc_lh"],
                              hist_sc=params["contrib_sc_h2"],
                              hist_oc=params["contrib_oc_h2"])
    compute_single_class_likelihood(contribs_path=str(os.path.join(experiment_dir, contrib_path_lh)),
                                    hist_path=str(os.path.join(experiment_dir, hist_path_2)),
                                    save_path=str(os.path.join(experiment_dir, lh_path_2)))
    # endregion
