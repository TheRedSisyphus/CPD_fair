import argparse
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt

from config.logger import create_logger, logging_levels
from config.parameters import hist_header, freq_pos, sigma_lb_pos, sigma_ub_pos, hist_node_id_pos
from src.readers import file_reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-h0", "--hist_file_0", required=True, type=str, help="Path of the first hist file")
    parser.add_argument("-h1", "--hist_file_1", required=False, type=str, help="Path of the second hist file")
    parser.add_argument("-n", "--nodes_id", required=False, type=str, nargs='+',
                        help="Plot histogram for selected node id")
    parser.add_argument("-s", "--save_dir", required=True, type=str, help="Where to save the plot")
    parser.add_argument("-log", "--log_level", required=False, type=str, default='warning',
                        help="Logging level, useful for tests or debug")
    args = parser.parse_args()

    logger = create_logger(name=os.path.basename(__file__), level=logging_levels[args.log_level])

    hist_0 = file_reader(path=args.hist_file_0, header=hist_header)
    if args.hist_file_1:
        hist_1 = file_reader(path=args.hist_file_1, header=hist_header)

    if args.nodes_id is None:
        nodes_id = list(set([line[hist_node_id_pos] for line in hist_0]))
        logger.warning('Detected more than 150 nodes in hist files. Plotting may take lot of time.')
    else:
        nodes_id = args.nodes_id

    for node_id in nodes_id:
        node_hist_0 = [line for line in hist_0 if line[hist_node_id_pos] == node_id]
        if args.hist_file_1:
            node_hist_1 = [line for line in hist_1 if line[hist_node_id_pos] == node_id]
            if len(node_hist_1) <= 0:
                logger.warning('Empty histograms')

        if len(node_hist_0) <= 0:
            logger.warning('Empty histograms')
        else:
            nbr_inputs_h_0 = sum([int(line[freq_pos]) for line in node_hist_0])
            if args.hist_file_1:
                nbr_inputs_h_1 = sum([int(line[freq_pos]) for line in node_hist_1])

            center_bins_0 = [(float(line[sigma_lb_pos]) + float(line[sigma_ub_pos])) / 2 for line in node_hist_0]
            width_0 = float(node_hist_0[0][sigma_ub_pos]) - float(node_hist_0[0][sigma_lb_pos])
            height_0 = [int(line[freq_pos]) for line in node_hist_0]
            if args.hist_file_1:
                center_bins_1 = [(float(line[sigma_lb_pos]) + float(line[sigma_ub_pos])) / 2 for line in node_hist_1]
                width_1 = float(node_hist_1[0][sigma_ub_pos]) - float(node_hist_1[0][sigma_lb_pos])
                height_1 = [int(line[freq_pos]) for line in node_hist_1]

            assert width_0 > 0
            assert len(center_bins_0) == len(height_0)
            assert all([int(h) > 0 for h in height_0])

            if args.hist_file_1:
                assert width_1 > 0
                assert len(center_bins_1) == len(height_1)
                assert all([int(h) > 0 for h in height_1])

            fig, ax = plt.subplots()
            fig.set_size_inches(12, 7)
            ax.set_title(f'Histograms of activation levels of node {node_id}')
            ax.set_xlabel('Activation levels (absolute units)')
            ax.set_ylabel('Number of input')

            ax.bar(x=center_bins_0, height=height_0, width=width_0,
                   alpha=0.3, color='blue', edgecolor='black', label=str(os.path.basename(args.hist_file_0)))

            if args.hist_file_1:
                ax.bar(x=center_bins_1, height=height_1, width=width_1,
                       alpha=0.3, color='red', edgecolor='black', label=str(os.path.basename(args.hist_file_1)))

            ax.legend()
            ax.grid()
            save_path_fig = Path(args.save_dir) / f'/histogram_plot_nid_{node_id}.png'
            fig.savefig(save_path_fig, dpi=300)
            plt.close(fig)

    sys.exit(0)
