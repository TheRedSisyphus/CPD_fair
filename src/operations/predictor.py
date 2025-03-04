import csv
import random
import warnings
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import config.parameters as p
from config.logger import create_logger
from config.parameters import EPSILON_PREC
from src.readers import file_reader
from src.utils import filter_indexes

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)


class Predictor(nn.Module):
    """Class used to make predictions on a dataset"""

    def __init__(self,
                 dimensions: list[int] | OrderedDict | None,
                 criterion=None,
                 regist_act_level=False,
                 lr=None,
                 seed=987654321,
                 c_weight: list[float] = None):

        super().__init__()

        self.regist_act_level = regist_act_level  # If we register activation levels during inference

        # Default value for learning rate
        if lr is None:
            lr = 0.01

        # Structure is the list of dimensions (number of neuron) for each layer of the model
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
            self.load_state_dict(dimensions)

        if criterion is None:
            if c_weight is not None:
                # c_weight is the classes weights, converted to double
                criterion = nn.NLLLoss(weight=torch.tensor(c_weight).double())
            else:
                criterion = nn.NLLLoss()
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # Todo : regulation
        self.lambda_reg = None  # HP used for regulation
        self.regularization = None

    def activate_registration(self):
        self.regist_act_level = True

    def deactivate_registration(self):
        self.regist_act_level = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Inference function on a torch Tensor"""
        if self.regist_act_level:
            hidden_data = self.seq(x)
            return self.output(hidden_data), hidden_data
        else:
            hidden_data = self.seq(x)
            return self.output(hidden_data), None

    def train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train one epoch of the model"""
        self.train()
        running_loss = 0.0
        acc = 0.0
        for inputs, labels in train_loader:
            labels = labels.long()
            # Predict and compute loss
            pred, _ = self.forward(inputs)
            # Compute loss
            loss = self.criterion(pred, labels)
            if self.regularization == "L1":
                l1_norm = sum(param.abs().sum() for param in self.parameters())
                loss += self.lambda_reg * l1_norm
            elif self.regularization == "L2":
                l2_norm = sum(param.pow(2).sum() for param in self.parameters())
                loss += self.lambda_reg * l2_norm

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
        """Train the model (call train_one_epoch 'epochs' time). Plot accuracy and loss evolution"""
        logger.info("Training model...")
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

                    logger.debug(
                        f"Epoch {epoch} | Validation loss : {val_loss:.2f} | Accuracy : {val_acc * 100:.2f}%")

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

                save_path_fig = Path(save_path).with_suffix('.pdf')
                fig.savefig(save_path_fig, dpi=300, format='pdf')

    def test(self, loader: DataLoader) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Inference of the model on torch Dataloader"""
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs, act_level = self.forward(inputs)

                # argmax is the second value returned by torch.max
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.double()
                correct_pred_mask = torch.eq(predicted, labels)

                # Print accuracy on current loader
                logger.debug(
                    f'Accuracy on current loader is {float(((labels == predicted) * 1.0).mean()) * 100:.2f} %')

            return act_level, correct_pred_mask

    def get_accuracy(self, loader: DataLoader) -> float:
        """Return model accuracy on torch Dataloader"""
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs, _ = self.forward(inputs)

                # argmax is the second value returned by torch.max
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.double()

            return float(((labels == predicted) * 1.0).mean())

    def get_model_decision(self, data_loader: DataLoader) -> torch.Tensor:
        """Returns model score inferred on torch Dataloader"""
        self.eval()
        model_decision = torch.empty((0,))
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs, _ = self.forward(inputs)
                model_decision = torch.cat((model_decision, outputs), 0)

        return model_decision

    def get_activation_levels(self, data_loader: DataLoader) -> np.ndarray:
        """Return activation levels on torch Dataloader"""
        self.activate_registration()
        levels_list = []
        levels_loader, _ = self.test(data_loader)
        levels_loader = levels_loader.numpy()
        levels_loader = np.round(levels_loader, decimals=EPSILON_PREC)
        levels_list.append(levels_loader)

        levels = np.concatenate(levels_list, axis=0)

        self.deactivate_registration()
        return levels

    def save_activation_levels(self,
                               index_path: Path,
                               data_loader: DataLoader,
                               save_path: Path,
                               layer_id: int,
                               set_name: str | None = None,
                               correct: str | bool | None = None,
                               output_class: str | list[str] | None = None,
                               sensitive_class: str | list[str] | None = None,
                               rand: bool = False):
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
        :param rand: If true, select two randoms part of the contributions (only use for plot node hist)
        :return: Write in file a four column document : 'inputId', 'layerId', 'nodeId', 'nodeContrib'
        """
        act_levels = self.get_activation_levels(data_loader)
        if len(act_levels.shape) != 2:
            raise ValueError('ERROR write_contribs: levels must be of dimension 2')

        indexes = file_reader(path=index_path, header=p.indexes_header)

        act_levels_index = filter_indexes(indexes, set_name=set_name)
        if len(act_levels_index) != act_levels.shape[0]:
            raise ValueError(
                f"Train loader et indexes have different size : {act_levels.shape[0]} and {act_levels_index}")

        id_list = filter_indexes(indexes,
                                 set_name=set_name,
                                 correct=correct,
                                 output_class=output_class,
                                 sensitive_class=sensitive_class)

        if rand:
            random.shuffle(id_list)
            id_list_a = id_list[0::2]
            id_list_b = id_list[1::2]

            with open(save_path.with_name("random_section_a").with_suffix('.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(p.contribs_header)

                for i, input_id in enumerate(act_levels_index):
                    if input_id in id_list_a:
                        for node_id in range(act_levels.shape[1]):
                            node_contrib = act_levels[i][node_id].item()
                            writer.writerow([input_id, layer_id, node_id, f'{node_contrib:.10f}'])

            with open(save_path.with_name("random_section_b").with_suffix('.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(p.contribs_header)

                for i, input_id in enumerate(act_levels_index):
                    if input_id in id_list_b:
                        for node_id in range(act_levels.shape[1]):
                            node_contrib = act_levels[i][node_id].item()
                            writer.writerow([input_id, layer_id, node_id, f'{node_contrib:.10f}'])

        else:
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(p.contribs_header)

                for i, input_id in enumerate(act_levels_index):
                    if input_id in id_list:
                        for node_id in range(act_levels.shape[1]):
                            node_contrib = act_levels[i][node_id].item()
                            writer.writerow([input_id, layer_id, node_id, f'{node_contrib:.10f}'])


def load_model(model_path: Path) -> OrderedDict:
    """From model save path (.pt file), get the model parameters"""
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
