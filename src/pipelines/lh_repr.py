import json
from pathlib import Path
from typing import Optional

import matplotlib.ticker as tick
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import config.parameters as p
from config.logger import create_logger
from src.readers import file_reader
from src.utils import parse_args, read_parameters


def read_lh_files(file_x: Path,
                  file_y: Path) -> tuple[tuple[list[int], list[float]], tuple[list[int], list[float]]] | None:
    """
    Read two likelihood files, and return for each the list of the id and the values
    :param file_x: First file
    :param file_y: Second file
    :return: A tuple (id_x, x_data), (id_y, y_data), where id_x and id_y are id list of the two files and
    x_data and y_data the CPL values for two files
    """
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
    # Since we work we normalized CPD values, we only consider positive values
    if any([y < 0 for y in y_1]) or any([y < 0 for y in y_2]):
        logger.warning(
            "get_area function is not designed to work with negative values. Please reconsider curves")
    # Check if data point of CPD curves are evenly spaced
    if not np.all(
            np.isclose(np.diff(x_1), np.full(shape=np.diff(x_1).shape, fill_value=np.diff(x_1)[0]), atol=p.EPSILON_PREC,
                       rtol=0.)):
        raise ValueError(f"X-coordinates are not evenly spaced {np.diff(x_1)}")
    # Check if data point of CPD curves are evenly spaced
    if not np.all(
            np.isclose(np.diff(x_2), np.full(shape=np.diff(x_2).shape, fill_value=np.diff(x_2)[0]), atol=p.EPSILON_PREC,
                       rtol=0.)):
        raise ValueError(f"X-coordinates are not evenly spaced {np.diff(x_2)}")

    dx_1 = float(np.diff(x_1)[0])
    dx_2 = float(np.diff(x_2)[0])
    area_1 = np.trapz(y_1, dx=dx_1)
    area_2 = np.trapz(y_2, dx=dx_2)
    return round(area_1 - area_2, p.EPSILON_PREC)


def plot_cpl_2d(file_x: Path,
                file_y: Path,
                save_dir: Path,
                color_bar: bool,
                title: str,
                axis_x_name: str,
                axis_y_name: str,
                axis_min: Optional[float] = None,
                axis_max: Optional[float] = None) -> str:
    """Create and save the scatter plot of CPD values in 2D"""
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
    scatter_save_path = save_dir / 'scatter.pdf'
    fig.savefig(scatter_save_path, dpi=300, format='pdf')

    # endregion

    # region Exporting coordinates and logging proportions
    json_data = [[int(id_x[i]), x_data[i], y_data[i]] for i in range(len(id_x))]

    with open(file=save_dir / 'coordinates.json', mode='w') as out_file:
        json.dump(json_data, out_file)

    closer_to_y = sum([elem[0] < elem[1] for elem in zip(x_data, y_data)])
    closer_to_x = sum([elem[0] >= elem[1] for elem in zip(x_data, y_data)])
    return (
        f"Le nombre de points tel que Y > X est : {closer_to_y} (sur un total de {len(x_data)} points), soit {(closer_to_y / len(x_data)) * 100:.2f}% des points.\n"
        f"Le nombre de points tel que Y ≤ X est : {closer_to_x} (sur un total de {len(x_data)} points), soit {(closer_to_x / len(x_data)) * 100:.2f}% des points.")
    # endregion


def cpd_fraction_pop(file_x: Path, file_y: Path, save_dir: Path, title: str, label_x: str, label_y: str) -> None:
    """Create and save the plot of CPD along fraction of population"""
    (_, x_data), (_, y_data) = read_lh_files(file_x, file_y)
    x_data.sort(reverse=True)
    y_data.sort(reverse=True)

    number_of_dots = 100  # High : smooth but more calculations, low : fewer calculations but rougher

    frac_pop_x = []
    frac_pop_y = []
    # For a certain amounts of CPD values between min and max,
    # we count percentage of profiles with smaller CPD of the CPD value considered
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
    plot_save_path = save_dir / 'plot.pdf'
    fig.savefig(plot_save_path, dpi=300, format='pdf')

    # endregion


def dual_cpd_fraction_pop(data_plain_left: Path,
                          data_plain_right: Path,
                          data_dashed_left: Path,
                          data_dashed_right: Path,
                          save_dir: Path,
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
    :return: Save a graph of four CPD distributions along fraction of respective population
    """
    # region read and treat data
    (_, data_left), (_, data_right) = read_lh_files(data_plain_left, data_plain_right)
    (_, data_left_dashed), (_, data_right_dashed) = read_lh_files(data_dashed_left, data_dashed_right)

    data_left.sort(reverse=False)
    data_right.sort(reverse=False)
    data_left_dashed.sort(reverse=False)
    data_right_dashed.sort(reverse=False)

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

    # For each percent between 0 and 1,
    # we search for the first CPD value such that the proportions of inputs with smaller CPD value is equal to percent
    # We repeat these operations for each series of data
    for percent in np.arange(0, 1 - 1 / len_plain, 1 / len_plain):
        for i, cpd_value in enumerate(data_left):
            if abs(i / len_plain - percent) < p.EPSILON:
                series_left.append((cpd_value-min(data_left)) / (max_left-min(data_left)))  # Normalization
                break

        for i, cpd_value in enumerate(data_right):
            if abs(i / len_plain - percent) < p.EPSILON:
                series_right.append((cpd_value - min(data_right)) / (max_right-min(data_right)))
                break

    for percent in np.arange(0, 1 - 1 / len_dashed, 1 / len_dashed):
        for i, cpd_value in enumerate(data_left_dashed):
            if abs(i / len_dashed - percent) < p.EPSILON:
                series_left_dashed.append((cpd_value - min(data_left_dashed)) / (max_left-min(data_left_dashed)))
                break

        for i, cpd_value in enumerate(data_right_dashed):
            if abs(i / len_dashed - percent) < p.EPSILON:
                series_right_dashed.append((cpd_value-min(data_right_dashed)) / (max_right-min(data_right_dashed)))
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
    ax.set_ylim(0, max(series_right + series_right_dashed) * 1.1)  # *1.1 to make graph readable

    # Second Y-axis
    sec_ax = ax.secondary_yaxis('right')
    sec_ax.set_ylabel(label_ax_right, color='green')
    sec_ax.set_ylim(0, max(series_left + series_left_dashed) * 1.1)  # *1.1 to make graph readable

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

    exp_nbr = save_dir.parent.parent.stem
    plot_save_path = save_dir / ('dual_plot_' + str(exp_nbr) + '_' + '.pdf')
    fig.savefig(plot_save_path, dpi=300, format='pdf')

    # endregion

    # region calculate areas

    curve_plain_left = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_left)), series_left)]
    curve_dashed_left = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_left_dashed)), series_left_dashed)]

    curve_plain_right = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_right)), series_right)]
    curve_dashed_right = [(x, y) for (x, y) in zip(np.linspace(0, 100, len(series_right_dashed)), series_right_dashed)]

    return (
        f"The area between the curves {data_plain_left.stem} and {data_dashed_left.stem} is {get_area(curve_plain_left, curve_dashed_left)}.\n"
        f"The area between the curves {data_plain_right.stem} and {data_dashed_right.stem} is {get_area(curve_plain_right, curve_dashed_right)}")
    # endregion


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
    params = read_parameters(args.param,
                             "file_x",
                             "file_y",
                             "plot_color_bar",
                             "plot_label_x",
                             "plot_label_y",
                             "title")

    # endregion

    plot_infos = plot_cpl_2d(file_x=Path(args.param).parents[1] / params["file_x"],
                             file_y=Path(args.param).parents[1] / params["file_y"],
                             save_dir=experiment_dir,
                             color_bar=params["plot_color_bar"],
                             title=params["title"],
                             axis_x_name=params["plot_label_x"],
                             axis_y_name=params["plot_label_y"],
                             axis_min=params.get("plot_axis_min_dim"),
                             axis_max=params.get("plot_axis_max_dim"))

    logger.info(plot_infos)

    cpd_fraction_pop(file_x=Path(args.param).parents[1] / params["file_x"],
                     file_y=Path(args.param).parents[1] / params["file_y"],
                     save_dir=experiment_dir,
                     title=params["title"],
                     label_x=params["plot_label_x"],
                     label_y=params["plot_label_y"])

    if params.get("data_plain_left") and params.get("data_plain_right") and params.get(
            "data_dashed_left") and params.get("data_dashed_right"):
        dual_plot_info = dual_cpd_fraction_pop(data_plain_left=experiment_dir.parents[1] / params["data_plain_left"],
                                               data_plain_right=experiment_dir.parents[1] / params["data_plain_right"],
                                               data_dashed_left=experiment_dir.parents[1] / params["data_dashed_left"],
                                               data_dashed_right=experiment_dir.parents[1] / params["data_dashed_right"],
                                               save_dir=experiment_dir,
                                               label_ax_left=params["label_ax_left"],
                                               label_ax_right=params["label_ax_right"],
                                               label_plain=params["label_plain"],
                                               label_dashed=params["label_dashed"],
                                               title=params["dual_plot_title"])

        logger.info(dual_plot_info)
