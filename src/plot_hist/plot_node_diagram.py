import sys
from math import sqrt
from pathlib import Path
from statistics import pstdev
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from numpy import ndarray
from torch import nn

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import load_model, get_model_structure, Predictor
from src.readers import file_reader
from src.utils import parse_args, read_parameters

# layerId, nodeId, binId, lb, ub, freq
hist_type = list[list[int, int, int, float, float, int]]
# Keys : node_id, values: histogram
table_hist_type = dict[int, hist_type]
# node_Id, avg, avg-std, avg + std
hist_info_type = list[int, float, float, float]
array_hist_info_type = list[hist_info_type]
magnitude_type = tuple[ndarray, ndarray, ndarray], tuple[ndarray, ndarray, ndarray]


def get_linear_layers(predictor: Predictor):
    all_layers = []

    def remove_sequential(net):
        for layer in net.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(layer)
            if not list(layer.children()):  # if leaf node, add it to list
                all_layers.append(layer)

    remove_sequential(predictor.output)

    return [layer for layer in all_layers if isinstance(layer, nn.Linear)]


def get_table_histo(contribs_file: Path, model_path: Path) -> table_hist_type:
    # Getting model structure
    model_param = load_model(model_path)
    _, structure, _ = get_model_structure(model_param)
    if len(structure) < 1:
        raise ValueError(f'ERROR get_histograms: structure of the model is too short')
    layer_id = len(structure) - 1
    # Reading contributions
    contribs = file_reader(contribs_file, header=p.contribs_header)

    # Initialize return value
    table_hist = {}

    # Iterate through nodeId and construct histogram
    for node_id in range(int(structure[layer_id])):
        histo_node = []
        try:
            contribs_node_id = [float(c[p.node_contrib_pos]) for c in contribs if
                                (c[p.contrib_layer_id_pos] == str(layer_id)) and (
                                            c[p.contrib_node_id_pos] == str(node_id))]
        except ValueError:
            raise ValueError(f'ERROR get_table_histo : impossible to convert some contrib to float in {contribs_file}')

        if len(contribs_node_id) <= 0:
            raise ValueError(f"ERROR get_table_histo: Empty file at {contribs_file}")

        sigma = pstdev(contribs_node_id)
        if sigma < p.EPSILON:
            raise ValueError(
                f'ERROR get_table_histo: Invalid distribution for node {node_id} of layer {layer_id}. The standard deviation is {sigma}'
            )
        # There are math.ceil((levels.max()-levels.min())/sigma) bins
        min_c = np.min(contribs_node_id)
        max_c = np.max(contribs_node_id)
        hist_node_id, bins_node_id = np.histogram(contribs_node_id, bins=np.arange(min_c, max_c + sigma, sigma))
        prev_bin = bins_node_id[0]
        for bin_id, bin_h in enumerate(bins_node_id[1:]):
            if hist_node_id[bin_id] != 0:
                histo_node.append([layer_id, node_id, bin_id, prev_bin, bin_h, hist_node_id[bin_id]])
            prev_bin = bin_h

        table_hist[node_id] = histo_node

    return table_hist


def get_histo_info(histo: hist_type) -> hist_info_type:
    avg = 0
    var = 0
    nbr_input = 0
    node_id = histo[0][1]
    for bin_h in histo:
        bin_center = (bin_h[3] + bin_h[4]) / 2
        avg += bin_center * bin_h[5]
        nbr_input += bin_h[5]
    avg = avg / nbr_input

    for bin_h in histo:
        bin_center = (bin_h[3] + bin_h[4]) / 2
        var += bin_h[5] * (bin_center - avg) * (bin_center - avg)
    var = var / nbr_input
    std = sqrt(var)
    return [node_id, avg, avg - std, avg + std]


def get_list_hist_info(table_histo: table_hist_type) -> array_hist_info_type:
    array_info_histo = []
    for node_id, histo in table_histo.items():
        histo_info = get_histo_info(histo)
        array_info_histo.append(histo_info)

    return array_info_histo


def get_last_hidden_layer_weights(model_path: Path) -> magnitude_type:
    parameters = load_model(model_path)
    predictor = Predictor(dimensions=parameters)
    last_hidden_layer = [layer for layer in predictor.output if isinstance(layer, nn.Linear)][0]
    weight, _ = last_hidden_layer.parameters()
    weight = weight.detach().numpy()
    if weight.shape[0] != 2:
        raise ValueError('ERROR get_last_hidden_layer_weights: model is not binary classifier')
    else:
        weight_class_0 = weight[0]
        order_0 = np.argsort(weight_class_0)[::-1]

    return order_0, weight_class_0


def plot_node(first_array_hist_info: array_hist_info_type,
              second_array_hist_info: array_hist_info_type,
              save_dir: Path,
              weights: Optional[np.ndarray] = None,
              min_w: Optional[float] = None,
              max_w: Optional[float] = None,
              legend_a: str = '',
              legend_b: str = '',
              y_min: Optional[float] = None,
              y_max: Optional[float] = None, ):
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 7)
    if y_min is None or y_max is None:
        y_min = min(
            [hist[2] for hist in first_array_hist_info] + [hist[2] for hist in second_array_hist_info])
        y_max = max(
            [hist[3] for hist in first_array_hist_info] + [hist[3] for hist in second_array_hist_info])
    y_axis_margin = .15
    x_min = 0
    x_max = len(first_array_hist_info)
    # print(x_min, x_max, y_min, y_max)
    ax.set(xlim=(x_min, x_max), ylim=(y_min - abs(y_min * y_axis_margin), y_max + abs(y_max * y_axis_margin)))
    if weights is not None:
        ax_mag = ax.twinx()
        ax_mag.set_ylabel('Weights of neuron', color='green')
        ax_mag.set_xlim(x_min, x_max)
        if min_w is not None and max_w is not None:
            ax_mag.set_ylim(min_w - abs(min_w * y_axis_margin), max_w + abs(max_w * y_axis_margin))
        else:
            min_w = np.min(weights)
            max_w = np.max(weights)
            ax_mag.set_ylim(min_w - abs(min_w * y_axis_margin), max_w + abs(max_w * y_axis_margin))
    # Axis, legend and title
    ax.set_xlabel('Node ID of the last hidden layer')
    ax.set_ylabel('Activation levels')

    bin_width = 1
    rect_width = bin_width / 3

    # Add another x_ticks to avoid truncation of bins
    x_ticks = [i + rect_width for i in range(x_max)] + [x_max + rect_width]
    x_labels = [str(line[0]) for line in first_array_hist_info] + ['']
    ax.set_xticks(ticks=x_ticks, labels=x_labels, rotation=90)

    legend_elements = [Patch(facecolor='cyan', edgecolor='black', lw=0.75, label=legend_a),
                       Patch(facecolor='orange', edgecolor='black', lw=0.75, label=legend_b)]
    if weights is not None:
        legend_elements.append(Line2D([0], [0], color='green', label='Weights', alpha=0.3))
    ax.legend(handles=legend_elements, loc="best")

    # Plots
    if weights is not None:
        ax_mag.plot(weights, color='green', alpha=0.3)

    for i in range(x_max):
        _, avg_1, avg_min_std_1, avg_plus_std_1 = first_array_hist_info[i]
        _, avg_2, avg_min_std_2, avg_plus_std_2 = second_array_hist_info[i]
        first_height = avg_plus_std_1 - avg_min_std_1
        second_height = avg_plus_std_2 - avg_min_std_2
        # Rectangle of hist 0
        ax.add_patch(
            Rectangle(xy=(i, avg_min_std_1), width=rect_width, height=first_height, facecolor='cyan',
                      edgecolor='black', lw=0.25))

        # We use 0.01 to avoid overlapping between rectangles and average lines
        ax.plot((i + 0.01, i + rect_width - 0.01), (avg_1, avg_1), 'black', lw=0.75)
        # Rectangle of hist 1
        ax.add_patch(
            Rectangle(xy=(i + rect_width, avg_min_std_2), width=rect_width, height=second_height,
                      facecolor='orange',
                      edgecolor='black', lw=0.25))
        ax.plot((i + rect_width + 0.01, i + 2 * rect_width - 0.01), (avg_2, avg_2), 'black', lw=0.75)

    # Saving plot
    plot_index = 0
    save_path_fig = save_dir / f'plot_{plot_index}.png'
    while save_path_fig.exists():
        plot_index += 1
        save_path_fig = save_dir / f'plot_{plot_index}.png'
    fig.savefig(save_path_fig, dpi=300)


def plot_diagram(first_array_hist_info: array_hist_info_type,
                 second_array_hist_info: array_hist_info_type,
                 save_dir: Path,
                 magnitude: Optional[tuple[np.ndarray, np.ndarray]] = None,
                 legend_a: str = '',
                 legend_b: str = '',
                 node_per_graph: Optional[int] = None):
    nbr_node = len(first_array_hist_info)
    if nbr_node != len(second_array_hist_info):
        raise ValueError('ERROR plot_diagram: both array_hist_info should have the same nodeId key')
    if node_per_graph is not None:
        if not (1 < node_per_graph < nbr_node):
            raise ValueError(f'ERROR plot_diagram: node_per_graph argument must be between 2 and {nbr_node}')
        nbr_graph = nbr_node // node_per_graph
        if nbr_node % node_per_graph != 0:
            logger.warning('node_per_graph argument is not multiple of number of node')

        for i in range(nbr_graph):
            y_min = min(min([line[2] for line in first_array_hist_info]),
                        min([line[2] for line in second_array_hist_info]))
            y_max = max(max([line[3] for line in first_array_hist_info]),
                        max([line[3] for line in second_array_hist_info]))
            if magnitude is None:
                weights, min_w, max_w = None, None, None
                portion_first_array_hist_info = first_array_hist_info
                portion_second_array_hist_info = second_array_hist_info
            else:
                order, weights = magnitude
                min_w, max_w = np.min(weights), np.max(weights)
                sorted_first_array_hist_info = [first_array_hist_info[i] for i in order]
                sorted_second_array_hist_info = [second_array_hist_info[i] for i in order]
                portion_first_array_hist_info = sorted_first_array_hist_info[
                                                i * node_per_graph:(i + 1) * node_per_graph]
                portion_second_array_hist_info = sorted_second_array_hist_info[
                                                 i * node_per_graph:(i + 1) * node_per_graph]
                weights = [weights[i] for i in order]
                weights = weights[i * node_per_graph:(i + 1) * node_per_graph]

            plot_node(first_array_hist_info=portion_first_array_hist_info,
                      second_array_hist_info=portion_second_array_hist_info,
                      save_dir=save_dir,
                      weights=weights,
                      min_w=min_w,
                      max_w=max_w,
                      legend_a=legend_a,
                      legend_b=legend_b,
                      y_min=y_min,
                      y_max=y_max)

    else:
        y_min = min(min([line[2] for line in first_array_hist_info]), min([line[2] for line in second_array_hist_info]))
        y_max = max(max([line[3] for line in first_array_hist_info]), max([line[3] for line in second_array_hist_info]))
        if magnitude is None:
            weights, min_w, max_w = None, None, None
            sorted_first_array_hist_info = first_array_hist_info
            sorted_second_array_hist_info = second_array_hist_info
        else:
            order, weights = magnitude
            min_w, max_w = np.min(weights), np.max(weights)
            sorted_first_array_hist_info = [first_array_hist_info[i] for i in order]
            sorted_second_array_hist_info = [second_array_hist_info[i] for i in order]
            weights = [weights[i] for i in order]

        plot_node(first_array_hist_info=sorted_first_array_hist_info,
                  second_array_hist_info=sorted_second_array_hist_info,
                  save_dir=save_dir,
                  weights=weights,
                  min_w=min_w,
                  max_w=max_w,
                  legend_a=legend_a,
                  legend_b=legend_b,
                  y_min=y_min,
                  y_max=y_max)


if __name__ == '__main__':
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
    params = read_parameters(args.param,
                             "contribs_a",
                             "contribs_b",
                             "legend_a",
                             "legend_b")

    model_p = experiment_dir.parent / p.data_dir_name / p.model_path

    # region Not random
    array_hist_a = get_table_histo(experiment_dir / params["contribs_a"], model_p)
    array_hist_b = get_table_histo(experiment_dir / params["contribs_b"], model_p)

    info_array_hist_a = get_list_hist_info(array_hist_a)
    info_array_hist_b = get_list_hist_info(array_hist_b)

    ord_0, weight_0 = get_last_hidden_layer_weights(model_p)

    plot_diagram(first_array_hist_info=info_array_hist_a,
                 second_array_hist_info=info_array_hist_b,
                 save_dir=experiment_dir,
                 magnitude=(ord_0, weight_0),
                 legend_a=params["legend_a"],
                 legend_b=params["legend_b"])

    # endregion
    # region random
    array_hist_a = get_table_histo(experiment_dir / "random_section_a.csv", model_p)
    array_hist_b = get_table_histo(experiment_dir / "random_section_b.csv", model_p)

    info_array_hist_a = get_list_hist_info(array_hist_a)
    info_array_hist_b = get_list_hist_info(array_hist_b)

    ord_0, weight_0 = get_last_hidden_layer_weights(model_p)

    plot_diagram(first_array_hist_info=info_array_hist_a,
                 second_array_hist_info=info_array_hist_b,
                 save_dir=experiment_dir,
                 magnitude=(ord_0, weight_0),
                 legend_a="Histogram of random section A",
                 legend_b="Histogram of random section B")
    # endregion

    sys.exit(0)
