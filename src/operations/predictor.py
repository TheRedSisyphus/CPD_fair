import csv
import os
import warnings
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config.parameters as p
from config.logger import create_logger
from config.parameters import EPSILON_PREC
from src.readers import file_reader


def filter_indexes(indexes_array: list[list[str]],
                   set_name: str | None,
                   correct: str | bool | None,
                   output_class: str | list[str] | None,
                   sensitive_class: str | list[str] | None) -> list[int]:
    """
    :param indexes_array: Unfiltered indexes to read
    :param set_name: None, '', 'train', 'test', 'valid'. The name of the set to filter
    :param correct: None, '', 'true',True, 'false', False. Filter input where model prediction is correct/incorrect
    :param output_class: Output class to filter. If output_class is a list, filter all input that have oc in this list
    :param sensitive_class: Sensitive class to filter.
    If sensitive_class is a list, filter all input that have sc in this list
    :return: A list of ID of input filtered
    """
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)
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


class Predictor(nn.Module):
    def __init__(self,
                 dimensions: list[int] | OrderedDict | None,
                 criterion=nn.NLLLoss(),
                 regist_act_level=False,
                 lr=None,
                 seed=987654321):

        super().__init__()

        self.regist_act_level = regist_act_level

        # Default value
        if lr is None:
            lr = 0.01

        if isinstance(dimensions, OrderedDict):
            self.structure = get_model_structure(dimensions)
        elif isinstance(dimensions, list):
            if not all([isinstance(elem, int) for elem in dimensions]):
                raise TypeError(
                    'ERROR class Predictor: dimensions arg must be either list of int, or Pytorch params (OrderedDict)')
            if len(dimensions) <= 2:
                raise ValueError('ERROR class Predictor: dimensions arg is too short (must be at least 3 elements)')

            self.structure = dimensions[0], dimensions[1:-1], dimensions[-1]
        else:
            raise TypeError(
                'ERROR class Predictor: dimensions arg must be either list of int, or Pytorch params (OrderedDict)')
        # Setting seed for model weights initialisation
        torch.manual_seed(seed)
        dim = self.structure[0]
        seq = []
        for item in self.structure[1]:
            seq += [nn.Linear(dim, item, dtype=torch.float64), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item
        output_layers = [nn.Linear(dim, self.structure[2], dtype=torch.float64), nn.LogSoftmax(dim=1)]
        self.seq = nn.Sequential(*seq)
        self.output = nn.Sequential(*output_layers)

        if isinstance(dimensions, OrderedDict):
            self.load_state_dict(dimensions)  # Generate a UserWarning

        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def activate_registration(self):
        self.regist_act_level = True

    def deactivate_registration(self):
        self.regist_act_level = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.regist_act_level:
            hidden_data = self.seq(x)
            return self.output(hidden_data), hidden_data
        else:
            hidden_data = self.seq(x)
            return self.output(hidden_data), None

    def train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.train()
        running_loss = 0.0
        acc = 0.0
        for inputs, labels in train_loader:
            labels = labels.long()
            # Predict and compute loss
            pred, _ = self.forward(inputs)
            # Compute loss
            loss = self.criterion(pred, labels)
            # Clear out the gradients of all variables of the optimizer
            self.optimizer.zero_grad()
            # Compute gradient loss
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()

            running_loss += loss.item()
            acc += torch.eq(pred.argmax(1), labels).sum().item()

        return running_loss, acc

    def train_model(self,
                    epochs: int,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    save_path: str,
                    validation_rate: int = 5,
                    plot: bool = True) -> None:
        logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL, file_dir=os.path.dirname(save_path))
        if epochs <= 0:
            torch.save(
                {"state_dict": self.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict()},
                save_path)
        else:
            best_val_loss = np.inf
            nbr_input = len(train_loader.dataset)  # type: ignore
            plot_data = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
            epoch_of_best = 0
            for epoch in range(1, epochs + 1):
                # Validate model every X epochs
                if epoch % validation_rate == 0:
                    self.eval()
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            labels = labels.long()
                            # Predict and compute loss
                            pred, _ = self.forward(inputs)
                            # Compute loss
                            val_loss = self.criterion(pred, labels)
                            val_acc = torch.eq(pred.argmax(1), labels).sum().item() / len(
                                valid_loader.dataset)  # type: ignore
                    # * validation_rate : useful to plot
                    plot_data["val_loss"].append(val_loss)
                    plot_data["val_accuracy"].append(val_acc)

                    logger.debug(f"Epoch {epoch} | Validation loss : {val_loss:.2f} | Accuracy : {val_acc * 100:.2f}%")

                    if val_loss < best_val_loss:
                        torch.save(
                            {"state_dict": self.state_dict(),
                             "optimizer_state_dict": self.optimizer.state_dict()},
                            save_path)
                        best_val_loss = val_loss
                        epoch_of_best = epoch
                else:
                    training_loss, train_acc = self.train_one_epoch(train_loader)
                    plot_data["train_loss"].append(training_loss)
                    plot_data["train_accuracy"].append(train_acc / nbr_input)

                    logger.debug(
                        f'Epoch {epoch} | Training loss : {training_loss:.2f} | Accuracy : {(train_acc / nbr_input) * 100:.2f}%')
            # If the number of epoch is too small, we save last epoch
            if epochs < validation_rate:
                torch.save(
                    {"state_dict": self.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict()},
                    save_path)
                logger.info(
                    f"Best model saved with training accuracy {plot_data['train_accuracy'][-1] * 100:.2f}%")
            else:
                logger.info(
                    f"Best model saved with validation accuracy {plot_data['val_accuracy'][epoch_of_best // validation_rate - 1] * 100:.2f}%")

            if plot:
                # Plot
                fig, (ax1, ax2) = plt.subplots(2, sharex=True)
                fig.set_size_inches(12, 7)
                ax1.plot([x for x in range(1, epochs + 1) if x % validation_rate != 0], plot_data["train_loss"],
                         label="Training Loss")
                ax1.plot([x for x in range(1, epochs + 1) if x % validation_rate == 0], plot_data["val_loss"],
                         label="Validation Loss")
                ax1.set_title("Training Loss")
                ax1.legend()
                ax1.set_xlabel('Number of epochs')
                ax2.plot([x for x in range(1, epochs + 1) if x % validation_rate != 0], plot_data["train_accuracy"],
                         label="Training Accuracy")
                ax2.plot([x for x in range(1, epochs + 1) if x % validation_rate == 0], plot_data["val_accuracy"],
                         label="Validation Accuracy")
                ax2.set_title("Accuracy")
                ax2.legend()
                ax2.set_xlabel('Number of epochs')

                save_path_fig = os.path.join(os.path.split(save_path)[0],
                                             os.path.basename(save_path)[:-3] + '.png')
                fig.savefig(save_path_fig, dpi=300)

    def test(self, loader: DataLoader, log_path: str = None) -> tuple[torch.Tensor | None, torch.Tensor]:
        logger = None
        if log_path is not None:
            logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL,
                                   file_dir=os.path.dirname(log_path))
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs, act_level = self.forward(inputs)

                # argmax is the second value returned by torch.max
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.double()
                correct_pred_mask = torch.eq(predicted, labels)

                # Print accuracy on current loader
                if log_path is not None:
                    logger.debug(
                        f'Accuracy on current loader is {float(((labels == predicted) * 1.0).mean()) * 100:.2f} %')

            return act_level, correct_pred_mask

    def get_precision(self, loader: DataLoader) -> float:
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs, _ = self.forward(inputs)

                # argmax is the second value returned by torch.max
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.double()

            return float(((labels == predicted) * 1.0).mean())

    def get_model_decision(self, data_loader: DataLoader) -> torch.Tensor:
        self.eval()
        model_decision = torch.empty((0,))
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs, _ = self.forward(inputs)
                model_decision = torch.cat((model_decision, outputs), 0)

        return model_decision

    def get_activation_levels(self, data_loader: DataLoader) -> np.ndarray:
        self.activate_registration()
        levels_list = []
        levels_loader, _ = self.test(data_loader)
        levels_loader = levels_loader.numpy()
        levels_loader = np.round(levels_loader, decimals=EPSILON_PREC)
        levels_list.append(levels_loader)

        levels = np.concatenate(levels_list, axis=0)

        self.deactivate_registration()
        return levels

    def write_indexes(self, save_path: str, train_path: str, sn_path: str, test_path: str, target: str, pa_path: str):
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerow(['inputId', 'TrueClass', 'PredictedClass', 'SensitiveAttr', 'SetName'])
            data = pd.read_csv(test_path, index_col='inputId')

            input_id = data.index.tolist()
            true_class = data[target].astype(int).tolist()
            data_loader = get_data_loader(train_data_path=train_path,
                                          test_data_path=test_path,
                                          set_name_path=sn_path,
                                          target=target)

            set_name = pd.read_csv(sn_path, index_col='inputId').squeeze().tolist()
            _, pred_correct = self.test(data_loader['test_db']['all'], log_path=os.path.dirname(save_path))
            predicted_class = [int(not bool(pred_correct[i]) ^ bool(true_class[i])) for i in range(len(input_id))]

            protec_attr = pd.read_csv(pa_path, index_col='inputId').squeeze().astype(int).tolist()

            for i, input_id in enumerate(input_id):
                writer.writerow([input_id, true_class[i], predicted_class[i], protec_attr[i], set_name[i]])

    def save_activation_levels(self,
                               index_path: str,
                               data_loader: DataLoader,
                               save_path: str,
                               layer_id: int,
                               set_name: str | None = None,
                               correct: str | bool | None = None,
                               output_class: str | list[str] | None = None,
                               sensitive_class: str | list[str] | None = None):
        """
        Write node contrib file
        :param index_path: Path to where indexes file is stored
        :param data_loader: Dataloader of all data
        :param save_path: Path to save the file
        :param layer_id: Layer to consider (usually is the penultimate/second-to-last)
        :param set_name: To filter the set
        :param correct: To filter if the model prediction is right
        :param output_class: To filter output class
        :param sensitive_class: To filter sensitive class
        :return: Write in file a four column document : 'inputId', 'layerId', 'nodeId', 'nodeContrib'
        """
        act_levels = self.get_activation_levels(data_loader)
        if len(act_levels.shape) != 2:
            raise ValueError('ERROR write_contribs: levels must be of dimension 2')

        indexes = file_reader(path=index_path, header=p.indexes_header)

        id_list = filter_indexes(indexes,
                                 set_name=set_name,
                                 correct=correct,
                                 output_class=output_class,
                                 sensitive_class=sensitive_class)

        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerow(p.contribs_header)

            for input_id in id_list:
                for node_id in range(act_levels.shape[1]):
                    node_contrib = act_levels[input_id][node_id].item()
                    writer.writerow([input_id, layer_id, node_id, f'{node_contrib:.10f}'])


def load_model(model_path: str) -> OrderedDict:
    """From model save path (.pt file), get the model parameters"""
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL, file_dir=os.path.dirname(model_path))
    with warnings.catch_warnings(action="ignore"):  # Raises a FutureWarning
        best_model = torch.load(model_path, map_location=torch.device('cpu'))
    try:
        parameters = best_model['state_dict']
    except KeyError:
        try:
            parameters = best_model['fair_state_dict']
        except KeyError:
            parameters = best_model['standard_state_dict']
            logger.warning("Deprecated model")
    return parameters


def get_model_structure(model_param: OrderedDict) -> (int, list[int], int):
    """From model parameters get the structure of the Pytorch model (input, structure, output)"""
    struct = [list(v.shape) for v in model_param.values()]
    struct = [elem for elem in struct if len(elem) > 1]
    input_dim = int(struct[0][1])
    output_dim = int(struct[-1][0])
    struct = [int(layer[1]) for layer in struct[1:]]
    return input_dim, struct, output_dim


def dataframe_to_loader(df: DataFrame, target: str, ) -> DataLoader:
    """Convert pandas dataframe to dataloader. Remove inputId column before processing"""
    features = torch.tensor(df.drop([target], axis=1).values)
    target_values = torch.tensor(df[target].values)
    dataset = TensorDataset(features, target_values)
    data_loader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False)
    return data_loader


def get_data_loader(train_data_path: str,
                    test_data_path: str,
                    set_name_path: str,
                    target: str) -> dict[str, dict[str, DataLoader]]:
    """
    To get the Dataloader of all preprocessed data, in a dict separated by set name (all, train, test, valid...)
    :return: A dict with key 'all', <other_set_name>. Each value corresponds to the loader of the key.
    Value of key 'all' is a dataloader of all data
    """

    set_name = pd.read_csv(set_name_path, index_col='inputId')
    set_name = set_name.squeeze()  # converting to pandas Series

    train_data = pd.read_csv(train_data_path, index_col='inputId')
    test_data = pd.read_csv(test_data_path, index_col='inputId')

    loaders = {"train_db": {"all": dataframe_to_loader(train_data, target=target)},
               "test_db": {"all": dataframe_to_loader(test_data, target=target)}}

    # Create smaller loaders for each set
    for set_n in set_name.unique():  # squeeze is to convert pd.Dataframe to pd.Series
        # Filter data of set set_n
        df_train = train_data[set_name == set_n]
        # Create data loader
        train_data_loader = dataframe_to_loader(df=df_train, target=target)
        loaders["train_db"][set_n] = train_data_loader

        df_test = test_data[set_name == set_n]
        test_data_loader = dataframe_to_loader(df=df_test, target=target)
        loaders["test_db"][set_n] = test_data_loader

    return loaders
