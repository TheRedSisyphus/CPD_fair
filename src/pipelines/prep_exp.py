import json
import os
from typing import Any

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor, get_data_loader
from src.operations.preprocess import generate_db
from src.utils import parse_args, get_target


# Todo : redo repr file
# Todo : rework logging in predictor.py
# Todo : compare old and new version of exp

def read_parameters(file: str) -> dict[str, Any]:
    """Read parameters file and return python dict with absolute path using config file"""
    base_dir = os.path.dirname(file)
    with open(file) as param:
        param_dict = json.load(param)

    for key in ['work_db', 'protec_attr', 'model']:
        if not param_dict.get(key):
            raise ValueError(f"Missing key in parameters : {key}")

    db_name = param_dict['work_db']
    param_dict['target'] = get_target(db_name)
    param_dict['work_db'] = os.path.join(p.db_refined_dir, param_dict['work_db'] + ".csv")
    param_dict["index_path"] = os.path.join(base_dir, p.indexes_path)

    param_dict['train']['save_path'] = os.path.join(base_dir, p.train_data_path)
    param_dict['test']['save_path'] = os.path.join(base_dir, p.test_data_path)

    # If we ignore protec attributes for test or train, then we ignore attr for both
    if bool(param_dict["train"].get("remove_protec_attr")) != bool(param_dict["train"].get("remove_protec_attr")):
        param_dict["train"]["remove_protec_attr"] = True
        param_dict["test"]["remove_protec_attr"] = True

    for key_model in ['dimensions', 'lr', 'epochs']:
        if not param_dict['model'].get(key_model):
            raise ValueError(f"Missing key in train parameters : {key_model}")

    param_dict['model']['save_path'] = os.path.join(base_dir, db_name + ".pt")
    param_dict['model']['train_path'] = param_dict['train']['save_path']
    param_dict['model']['test_path'] = param_dict['test']['save_path']
    param_dict['model']['set_name_path'] = os.path.join(os.path.dirname(param_dict['model']['train_path']),
                                                        p.set_name_path)
    param_dict['model']['pa_path'] = os.path.join(base_dir, p.protec_attr_path)

    with open(os.path.join(base_dir, "protocol_exp.txt"), mode='w') as protocol:
        protocol.write("=== GENERAL INFORMATION ==\n")
        protocol.write(f"WORK DATABASE : {db_name}\n"
                       f"TASK OF PREDICTION : {param_dict['target']}\n"
                       f"PROTECTED ATTRIBUTE : {param_dict["protec_attr"]}\n")
        # Train, test database
        if param_dict["train"].get("remove_protec_attr") is True:
            protocol.write("Model not aware of protected group")

        protocol.write("\n=== DATABASE INFORMATION ===\n")
        if param_dict["train"].get("treatment"):
            protocol.write(f"Train database is treated with : {param_dict["train"].get("treatment")}\n")
        if param_dict["test"].get("treatment"):
            protocol.write(f"Train database is treated with : {param_dict["test"].get("treatment")}\n")
        if not param_dict["train"].get("treatment") and not param_dict["test"].get("treatment"):
            protocol.write("Train and test are not treated\n")
        if param_dict["train"].get("treatment") == param_dict["test"].get("treatment"):
            protocol.write("(Train and test get same treatment)\n")
        # Model
        protocol.write("\n=== MODEL INFORMATION ===\n")
        protocol.write(f"Model dimensions are : {param_dict['model']["dimensions"]}\n")
        protocol.write(f"Model learning rate is : {param_dict['model']["lr"]}\n")
        protocol.write(f"Number of epochs for training model is : {param_dict['model']["epochs"]}\n")

    return param_dict


if __name__ == "__main__":
    args = parse_args()
    experiment_dir = os.path.dirname(args.param)
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL, file_dir=experiment_dir)

    params = read_parameters(args.param)

    # region DB TRAIN

    generate_db(work_db=params["work_db"],
                save_path=params["train"]["save_path"],
                target=params["target"],
                treatment_param=params["train"],
                desc_protec=params["protec_attr"],
                train=True)

    logger.info(f"Train database generated at {params["train"]["save_path"]}")

    # endregion

    # region MODEL TRAIN
    model = Predictor(dimensions=params["model"]["dimensions"], lr=params["model"]["lr"])

    load = get_data_loader(train_data_path=params["model"]["train_path"],
                           test_data_path=None,
                           set_name_path=params["model"]["set_name_path"],
                           target=params["target"])

    model.train_model(epochs=params["model"]["epochs"],
                      save_path=params["model"]["save_path"],
                      train_loader=load['train_db']['train'],
                      valid_loader=load['train_db']['valid'])

    logger.info(f"Model trained and saved at {params["model"]["save_path"]}")

    # endregion

    # region DB TEST

    generate_db(work_db=params["work_db"],
                save_path=params["test"]["save_path"],
                target=params["target"],
                treatment_param=params["test"],
                desc_protec=params["protec_attr"],
                train=False)

    logger.info(f"Train database generated at {params["test"]["save_path"]}")

    # endregion

    # region write indexes
    model.write_indexes(save_path=params["index_path"],
                        target=params["target"],
                        train_path=params["model"]["train_path"],
                        test_path=params["model"]["test_path"],
                        sn_path=params["model"]["set_name_path"],
                        pa_path=params["model"]["pa_path"])

    logger.info(f"Indexes generated at {params["index_path"]}")

    # endregion
