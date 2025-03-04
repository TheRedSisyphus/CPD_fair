from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

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


def plot_cpl_2d_all(data: pd.DataFrame,
                    file_m_lr: Path,
                    file_m_hr: Path,
                    file_f_lr: Path,
                    file_f_hr: Path,
                    save_dir: Path,
                    title: str,
                    axis_x_name: str,
                    axis_y_name: str,
                    axis_min: Optional[float] = None,
                    axis_max: Optional[float] = None):
    """Create and save the scatter plot of CPD values in 2D"""
    (id_m_data, x_m_data), (_, y_m_data) = read_lh_files(file_m_lr, file_m_hr)
    (id_f_data, x_f_data), (_, y_f_data) = read_lh_files(file_f_lr, file_f_hr)

    high_educ = data[data['education'] == 1.0].index
    low_educ = data[data['education'] != 1.0].index

    x_m_le = [elem for i, elem in enumerate(x_m_data) if id_m_data[i] in low_educ]
    x_m_he = [elem for i, elem in enumerate(x_m_data) if id_m_data[i] in high_educ]

    y_m_le = [elem for i, elem in enumerate(y_m_data) if id_m_data[i] in low_educ]
    y_m_he = [elem for i, elem in enumerate(y_m_data) if id_m_data[i] in high_educ]

    x_f_le = [elem for i, elem in enumerate(x_f_data) if id_f_data[i] in low_educ]
    x_f_he = [elem for i, elem in enumerate(x_f_data) if id_f_data[i] in high_educ]

    y_f_le = [elem for i, elem in enumerate(y_f_data) if id_f_data[i] in low_educ]
    y_f_he = [elem for i, elem in enumerate(y_f_data) if id_f_data[i] in high_educ]

    x_m_le = list(set(x_m_le))
    x_m_he = list(set(x_m_he))

    y_m_le = list(set(y_m_le))
    y_m_he = list(set(y_m_he))

    x_f_le = list(set(x_f_le))
    x_f_he = list(set(x_f_he))

    y_f_le = list(set(y_f_le))
    y_f_he = list(set(y_f_he))

    # region Creating and saving figure
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    ax.set_title(title, wrap=True)

    ax.set_xlabel(axis_x_name)
    ax.set_ylabel(axis_y_name)

    ax.scatter(x=x_m_le, y=y_m_le, color='blue', marker='v', label='Uneducated men', alpha=0.5)
    ax.scatter(x=x_m_he, y=y_m_he, color='blue', marker='^', label='Educated men', alpha=0.5)
    ax.scatter(x=x_f_le, y=y_f_le, color='red', marker='v', label='Uneducated women', alpha=0.5)
    ax.scatter(x=x_f_he, y=y_f_he, color='red', marker='^', label='Educated women', alpha=0.5)

    ax.set(xlim=(axis_min, axis_max), ylim=(axis_min, axis_max))

    ax.axline((0, 0), slope=1, color='black', linewidth=0.5)

    # Make graph square
    ax.set_aspect('equal', adjustable='box')
    # Grid
    plt.grid()
    # legend
    plt.legend(loc='upper right')
    # Saving plot
    scatter_save_path = save_dir / f'synthetic_{save_dir.parent.stem.split('_')[-1]}.pdf'
    fig.savefig(scatter_save_path, dpi=300, format='pdf')


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
    params = read_parameters(args.param,
                             "file_m_lr",
                             "file_m_hr",
                             "file_f_lr",
                             "file_f_hr",
                             "plot_label_x",
                             "plot_label_y",
                             "plot_axis_min",
                             "plot_axis_max",
                             "title")

    # endregion

    plot_cpl_2d_all(data=pd.read_csv(experiment_dir.parents[0] / "_data"/"data.csv", index_col='inputId'),
                    file_m_lr=experiment_dir.parents[0] / (params['file_m_lr']),
                    file_m_hr=experiment_dir.parents[0] / (params['file_m_hr']),
                    file_f_lr=experiment_dir.parents[0] / (params['file_f_lr']),
                    file_f_hr=experiment_dir.parents[0] / (params['file_f_hr']),
                    save_dir=experiment_dir,
                    axis_x_name=params['plot_label_x'],
                    axis_y_name=params['plot_label_y'],
                    title=params['title'],
                    axis_min=params['plot_axis_min'],
                    axis_max=params['plot_axis_max'])
