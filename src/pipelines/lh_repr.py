import json
import os
from typing import Optional, Any

import matplotlib.ticker as tick
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import config.parameters as p
from config.logger import create_logger
from src.readers import file_reader
from src.utils import parse_args


def read_parameters(file: str) -> dict[str, Any]:
    """Read parameters file and return python dict with absolute path using config file"""
    with open(file) as param:
        param_dict = json.load(param)

    for key in ["file_x",
                "file_y",
                "scatter_color_bar"]:
        if param_dict.get(key) is None:
            raise ValueError(f"Missing key in parameters : {key}")

    replaceable = [("plot_title", "scatter_title"), ("plot_label_x", "scatter_x_axis_name"),
                   ("plot_label_y", "scatter_y_axis_name")]  # Attr that can be interchanged

    for attr_1, attr_2 in replaceable:
        if param_dict.get(attr_1) is None and param_dict.get(attr_2) is None:
            raise ValueError(f"Missing parameters {attr_1} or {attr_2}")
        if attr_1 not in param_dict and attr_2 in param_dict:
            param_dict[attr_1] = param_dict[attr_2]
        if "scatter_title" not in param_dict and attr_1 in param_dict:
            param_dict[attr_2] = param_dict[attr_1]

    if "plot_title" not in param_dict and "scatter_title" in param_dict:
        param_dict["plot_title"] = param_dict["scatter_title"]
    if "scatter_title" not in param_dict and "plot_title" in param_dict:
        param_dict["scatter_title"] = param_dict["plot_title"]

    # Use scatter axis and title names for plot

    if param_dict.get("plot_label_x") is None:
        param_dict["plot_label_x"] = param_dict["scatter_x_axis_name"]
    if param_dict.get("plot_label_y") is None:
        param_dict["plot_label_y"] = param_dict["scatter_y_axis_name"]

    result_dir = os.path.dirname(os.path.dirname(file))

    param_dict["file_x"] = os.path.join(result_dir, param_dict["file_x"])
    param_dict["file_y"] = os.path.join(result_dir, param_dict["file_y"])

    exp_dir = os.path.dirname(result_dir)
    if param_dict.get("data_plain_left") is not None:
        param_dict["data_plain_left"] = os.path.join(exp_dir, param_dict["data_plain_left"])
    if param_dict.get("data_plain_right") is not None:
        param_dict["data_plain_right"] = os.path.join(exp_dir, param_dict["data_plain_right"])
    if param_dict.get("data_dashed_left") is not None:
        param_dict["data_dashed_left"] = os.path.join(exp_dir, param_dict["data_dashed_left"])
    if param_dict.get("data_dashed_right") is not None:
        param_dict["data_dashed_right"] = os.path.join(exp_dir, param_dict["data_dashed_right"])

    return param_dict


def read_lh_files(file_x: str,
                  file_y: str) -> tuple[tuple[list[int], list[float]], tuple[list[int], list[float]]] | None:
    (coordinates_x, coordinates_y) = file_reader(file_x, header=p.lh_header), file_reader(file_y, header=p.lh_header)

    try:
        id_x = [int(elem[p.input_id_pos]) for elem in coordinates_x]
        id_y = [int(elem[p.input_id_pos]) for elem in coordinates_y]
    except Exception:
        raise ValueError(
            f'ERROR plot_cpl_2d : impossible to convert to int some inputs of {file_x} or {file_y}')

    if id_x != id_y:
        raise ValueError(f'ERROR plot_cpl_2d : Input Id for files {file_x}, {file_y} must be the same')

    try:
        x_data = [float(elem[p.score_pos]) for elem in coordinates_x]
        y_data = [float(elem[p.score_pos]) for elem in coordinates_y]
    except Exception:
        raise ValueError(
            f'ERROR plot_cpl_2d : impossible to convert to float some elements of {file_x} or {file_y}')

    if not len(x_data):
        logger.warning(f"Files {file_x} and {file_y} are empty")
        return

    return (id_x, x_data), (id_y, y_data)


def get_area(curve_1: list[tuple[float, float]], curve_2: list[tuple[float, float]]) -> float:
    """Get absolute value of area between two curves. Use Trapezoidal Rule."""
    x_1 = [x for x, _ in curve_1]
    x_2 = [x for x, _ in curve_2]
    y_1 = [y for _, y in curve_1]
    y_2 = [y for _, y in curve_2]
    if any([y < 0 for y in y_1]) or any([y < 0 for y in y_2]):
        logger.warning(
            "get_area function is not designed to work with negative values. Please reconsider curves")
    if not np.all(
            np.isclose(np.diff(x_1), np.full(shape=np.diff(x_1).shape, fill_value=np.diff(x_1)[0]), atol=p.EPSILON_PREC,
                       rtol=0.)):
        raise ValueError(f"X-coordinates are not evenly spaced {np.diff(x_1)}")
    if not np.all(
            np.isclose(np.diff(x_2), np.full(shape=np.diff(x_2).shape, fill_value=np.diff(x_2)[0]), atol=p.EPSILON_PREC,
                       rtol=0.)):
        raise ValueError(f"X-coordinates are not evenly spaced {np.diff(x_2)}")

    dx_1 = float(np.diff(x_1)[0])
    dx_2 = float(np.diff(x_2)[0])
    area_1 = np.trapz(y_1, dx=dx_1)
    area_2 = np.trapz(y_2, dx=dx_2)
    return round(area_1 - area_2, p.EPSILON_PREC)


def plot_cpl_2d(file_x: str,
                file_y: str,
                save_dir: str,
                color_bar: bool,
                title: str,
                axis_x_name: str,
                axis_y_name: str,
                axis_min: Optional[float] = None,
                axis_max: Optional[float] = None) -> str:
    (id_x, x_data), (id_y, y_data) = read_lh_files(file_x, file_y)

    # region Creating and saving figure
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5.5) if color_bar else fig.set_size_inches(7, 7)
    ax.set_title(title, wrap=True)

    ax.set_xlabel(axis_x_name)
    ax.set_ylabel(axis_y_name)

    if color_bar:
        # Get intensity
        xy = np.vstack([x_data, y_data])
        z = gaussian_kde(xy)(xy)
        # Plot densest point last
        idx = z.argsort()
        x_data, y_data, z = np.array(x_data)[idx], np.array(y_data)[idx], z[idx]
        scat = ax.scatter(x=x_data, y=y_data, s=1.0, c=z, cmap='plasma')
        c_bar = fig.colorbar(scat, shrink=1, aspect=40, anchor=(1.0, 0.5))
        c_bar.set_ticks([])
    else:
        ax.scatter(x=x_data, y=y_data, s=1.0)

    # Set same limits for x-axis and y-axis
    if not isinstance(axis_min, type(axis_max)):
        raise ValueError(
            f"You must define a minimum AND a maximum for axis dimension, or set both to None, got {axis_min} and {axis_max}")

    if axis_min is None and axis_max is None:
        axis_min = 0
        axis_max = 1.1 * max(max(x_data), max(y_data))

    ax.set(xlim=(axis_min, axis_max), ylim=(axis_min, axis_max))

    ax.axline((0, 0), slope=1, color='black', linewidth=0.5)

    # Make graph square
    ax.set_aspect('equal', adjustable='box')
    # Grid
    plt.grid()
    # Saving plot
    scatter_save_path = os.path.join(save_dir, 'scatter')
    fig.savefig(scatter_save_path, dpi=300)

    # endregion

    # region Exporting coordinates and logging proportions
    json_data = [[int(id_x[i]), x_data[i], y_data[i]] for i in range(len(id_x))]

    with open(file=os.path.join(save_dir, 'coordinates.json'), mode='w') as out_file:
        json.dump(json_data, out_file)

    closer_to_y = sum([elem[0] < elem[1] for elem in zip(x_data, y_data)])
    closer_to_x = sum([elem[0] >= elem[1] for elem in zip(x_data, y_data)])
    return (
        f"Le nombre de points tel que Y > X est : {closer_to_y} (sur un total de {len(x_data)} points), soit {(closer_to_y / len(x_data)) * 100:.2f}% des points.\n"
        f"Le nombre de points tel que Y ≤ X est : {closer_to_x} (sur un total de {len(x_data)} points), soit {(closer_to_x / len(x_data)) * 100:.2f}% des points.")
    # endregion


def cpd_fraction_pop(file_x: str, file_y: str, save_dir: str, title: str, label_x: str, label_y: str) -> None:
    (_, x_data), (_, y_data) = read_lh_files(file_x, file_y)
    x_data.sort(reverse=True)
    y_data.sort(reverse=True)

    number_of_dots = 100  # High : smooth but more calculations, low : fewer calculations but rougher

    frac_pop_x = []
    frac_pop_y = []
    for cpd_value in np.linspace(min(x_data), max(x_data), number_of_dots):
        smaller_than = [elem for elem in x_data if elem > cpd_value]
        frac_pop_x.append(100 * len(smaller_than) / len(x_data))

    for cpd_value in np.linspace(min(y_data), max(y_data), number_of_dots):
        smaller_than = [elem for elem in y_data if elem > cpd_value]
        frac_pop_y.append(100 * len(smaller_than) / len(y_data))

    # region Creating and saving figure
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    ax.set_title(title, wrap=True)

    ax.set_xlabel("Normalized CPD")
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, _: f'{x / number_of_dots:.1f}'))
    ax.set_ylabel("Fraction of respective population")
    ax.yaxis.set_major_formatter(tick.PercentFormatter())

    ax.plot(frac_pop_x, label=label_x, color='blue')
    ax.plot(frac_pop_y, label=label_y, color='red')

    # Grid and legend
    plt.grid()
    plt.legend()
    # Saving plot
    plot_save_path = os.path.join(save_dir, 'plot')
    fig.savefig(plot_save_path, dpi=300)

    # endregion


def dual_cpd_fraction_pop(data_plain_left: str,
                          data_plain_right: str,
                          data_dashed_left: str,
                          data_dashed_right: str,
                          save_dir: str,
                          title: str = "",
                          label_ax_left: str = "",
                          label_ax_right: str = "",
                          label_plain: str = "",
                          label_dashed: str = "") -> str:
    """
    :param data_plain_left: First series of plain data, same pop as data_plain_1
    :param data_plain_right: Second series of plain data, same pop as data_plain_0
    :param data_dashed_right: First series of dashed data, same pop as data_dashed_1
    :param data_dashed_left: Second series of dashed data, same pop as data_dashed_0
    :param save_dir: Save path
    :param title: Title of the figure
    :param label_ax_left: Label for the left Y-axis
    :param label_ax_right: Label for the right Y-axis
    :param label_plain: Label for plain curves
    :param label_dashed: Label for dashed curves
    :return: Save a graph
    """
    # region read and treat data
    (_, data_left), (_, data_right) = read_lh_files(data_plain_left, data_plain_right)
    (_, data_left_dashed), (_, data_right_dashed) = read_lh_files(data_dashed_left, data_dashed_right)

    data_left.sort(reverse=True)
    data_right.sort(reverse=True)
    data_left_dashed.sort(reverse=True)
    data_right_dashed.sort(reverse=True)

    series_left = []
    series_left_dashed = []
    series_right = []
    series_right_dashed = []

    # len(data_left) == len(data_right) thanks to read_lh_files
    len_plain = len(data_left)
    len_dashed = len(data_left_dashed)

    # For normalization
    max_left = max(data_left + data_left_dashed)
    max_right = max(data_right + data_right_dashed)

    for percent in np.arange(0, 1 - 1 / len_plain, 1 / len_plain):
        for i, cpd_value in enumerate(data_left):
            if abs(i / len_plain - percent) < p.EPSILON:
                series_left.append(cpd_value / max_left)
                break

        for i, cpd_value in enumerate(data_right):
            if abs(i / len_plain - percent) < p.EPSILON:
                series_right.append(cpd_value / max_right)
                break

    for percent in np.arange(0, 1 - 1 / len_dashed, 1 / len_dashed):
        for i, cpd_value in enumerate(data_left_dashed):
            if abs(i / len_dashed - percent) < p.EPSILON:
                series_left_dashed.append(cpd_value / max_left)
                break

        for i, cpd_value in enumerate(data_right_dashed):
            if abs(i / len_dashed - percent) < p.EPSILON:
                series_right_dashed.append(cpd_value / max_right)
                break

    # endregion

    # region plot figure

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.set_title(title, wrap=True)

    # X-axis
    ax.set_xlabel("Fraction of respective population")
    ax.xaxis.set_major_formatter(tick.PercentFormatter())
    # Y-axis
    ax.set_ylabel(label_ax_left, color='red')
    ax.set_ylim(0, max(series_left + series_left_dashed) * 1.1)  # *1.1 to make graph readable

    # Second Y-axis
    sec_ax = ax.secondary_yaxis('right')
    sec_ax.set_ylabel(label_ax_right, color='green')
    sec_ax.set_ylim(0, max(series_right + series_right_dashed) * 1.1)  # *1.1 to make graph readable

    # Color axis and change width
    ax.spines['left'].set_color('red')
    ax.spines['right'].set_color('green')
    ax.spines[['left', 'right']].set_linewidth(2)

    # Plot left
    ax.plot(np.linspace(0, 100, len(series_left)), series_left, color='red')
    ax.plot(np.linspace(0, 100, len(series_left_dashed)), series_left_dashed, color='red', linestyle='dashed')

    # Plot right
    ax.plot(np.linspace(0, 100, len(series_right)), series_right, color='green')
    ax.plot(np.linspace(0, 100, len(series_right_dashed)), series_right_dashed, color='green', linestyle='dashed')

    # Custom legend
    legend_elements = [Line2D([0], [0], color='grey', lw=1, label=label_plain),
                       Line2D([0], [0], color='grey', lw=1, linestyle='dashed', label=label_dashed)]

    ax.legend(handles=legend_elements, loc='best')

    plt.grid()

    plot_save_path = os.path.join(save_dir, 'dual_plot')
    fig.savefig(plot_save_path, dpi=300)

    # endregion

    # region calculate areas

    curve_plain_left = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_left)), series_left)]
    curve_dashed_left = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_left_dashed)), series_left_dashed)]

    curve_plain_right = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_right)), series_right)]
    curve_dashed_right = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_right_dashed)), series_right_dashed)]

    return (
        f"The area between the curves {os.path.basename(data_plain_left)} and {os.path.basename(data_dashed_left)} is {get_area(curve_plain_left, curve_dashed_left)}.\n"
        f"The area between the curves {os.path.basename(data_plain_right)} and {os.path.basename(data_dashed_right)} is {get_area(curve_plain_right, curve_dashed_right)}")
    # endregion


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = os.path.dirname(args.param)
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL, file_dir=experiment_dir)
    # endregion

    params = read_parameters(args.param)

    plot_infos = plot_cpl_2d(file_x=params["file_x"],
                             file_y=params["file_y"],
                             save_dir=experiment_dir,
                             color_bar=params["scatter_color_bar"],
                             title=params["scatter_title"],
                             axis_x_name=params["scatter_x_axis_name"],
                             axis_y_name=params["scatter_y_axis_name"],
                             axis_min=params["scatter_axis_min_dim"],
                             axis_max=params["scatter_axis_max_dim"])

    logger.info(plot_infos)

    cpd_fraction_pop(file_x=params["file_x"],
                     file_y=params["file_y"],
                     save_dir=experiment_dir,
                     title=params["plot_title"],
                     label_x=params["plot_label_x"],
                     label_y=params["plot_label_y"])

    if params.get("data_plain_left") and params.get("data_plain_right") and params.get(
            "data_dashed_left") and params.get("data_dashed_right"):
        dual_plot_info = dual_cpd_fraction_pop(data_plain_left=params["data_plain_left"],
                                               data_plain_right=params["data_plain_right"],
                                               data_dashed_left=params["data_dashed_left"],
                                               data_dashed_right=params["data_dashed_right"],
                                               save_dir=experiment_dir,
                                               label_ax_left=params["label_ax_left"],
                                               label_ax_right=params["label_ax_right"],
                                               label_plain=params["label_plain"],
                                               label_dashed=params["label_dashed"],
                                               title=params["dual_plot_title"])

        logger.info(dual_plot_info)
