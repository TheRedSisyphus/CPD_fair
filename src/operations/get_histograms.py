import csv
import os
from statistics import pstdev

import numpy as np

import config.parameters as p
from config.logger import create_logger
from src.readers import file_reader

logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)


def construct_hist(layer_id: str,
                   node_id: str,
                   act_levels: list[str],
                   save_path: str) -> None:
    nbr_act_levels = len(act_levels)

    if nbr_act_levels <= 0 or '' in act_levels:
        raise ValueError(
            f'ERROR construct_hist: Activation levels not found for node {node_id} at layer {layer_id}')

    # Counting in the string the number of decimal digits. If higher than 10, raise error, if equal to -1, '.' is not found
    contrib_prec = [0 <= elem[::-1].find('.') <= p.EPSILON_PREC for elem in act_levels]
    if not all(contrib_prec):
        false_indexes = [i for i, elem in enumerate(contrib_prec) if elem is False]
        raise ValueError(
            f'ERROR construct_hist: contribs file contains act levels with a precision higher than {p.EPSILON}, or contribs that are not float (dot character not found). At line {false_indexes}')

    try:
        act_levels = [float(elem) for elem in act_levels]
    except ValueError:
        raise ValueError(f'ERROR construct_hist: Impossible to convert act levels to float, please check contribs file')

    sigma = pstdev(act_levels)
    sigma = round(sigma, p.EPSILON_PREC) if sigma >= p.EPSILON else 0.0
    max_ = max(act_levels)
    min_ = min(act_levels)
    logger.debug(f'Standard deviation is {sigma:.10f}')

    # Standard dev is neg (impossible)
    if sigma < -p.EPSILON:
        raise ValueError(
            f'ERROR construct_hist: Invalid distribution for node {node_id} of layer {layer_id}. The standard deviation is negative : {sigma}'
        )
    # We append file since the header is already written
    with open(save_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        # Standard dev is null (no variance)
        if -p.EPSILON <= sigma < p.EPSILON:
            str_single_value = f"{min_:.10f}"
            logger.warning(f'Distribution for node {node_id} of layer {layer_id} with null variance')
            logger.debug(
                f'Unique binId 0 of bounds ({str_single_value}, {str_single_value}), and frequency {nbr_act_levels}\n')
            csvwriter.writerow([layer_id, node_id, 0, str_single_value, str_single_value, nbr_act_levels])

        # Standard dev is pos (most of the case)
        else:
            # Step is sigma - LSP, to avoid that contributions fall on bin edges
            hist, bins = np.histogram(act_levels,
                                      bins=np.arange(start=min_, stop=max_ + 2 * sigma,
                                                     step=sigma - p.LOW_SMOOTHED_PROB))

            logger.debug(f'Min is {min_:.10f} et max is {max_:.10f}, step {sigma:.10f}')

            if not np.isclose(bins, np.round(bins, p.EPSILON_PREC), atol=p.EPSILON_PREC).all():
                raise ValueError(f'ERROR construct_hist: bins cannot be rounded to precision')

            bins = np.round(bins, p.EPSILON_PREC)

            step = bins[1] - bins[0]
            logger.debug(f'The step is {step:.10f}')

            if len(bins) < 2:
                raise ValueError(f'ERROR construct_hist: too few bins for node {node_id} of layer {layer_id}')
            prev_bin = bins[0]
            if step < p.EPSILON:
                raise ValueError(f'ERROR construct_hist: step is null for node {node_id} of layer {layer_id}')

            nbr_bins = len(bins[1:])
            logger.debug(f'There are {nbr_bins} bins.')
            if nbr_bins > p.POS_UNDEF_INT:
                raise ValueError(
                    f'ERROR construct_hist: There are {nbr_bins}, which is too high for the current precision. '
                    f'Hist file would be incorrect')
            # For verification purpose
            count = 0
            for i, b in enumerate(bins[1:]):
                logger.debug(
                    f'BinId {i} ({prev_bin:.10f}, {b:.10f}) has a frequency of {hist[i]}' + (
                        '\n' if (i == nbr_bins - 1) else ''))

                if hist[i] != 0:
                    count += hist[i]
                    csvwriter.writerow([layer_id,
                                        node_id,
                                        i,
                                        f'{prev_bin:.10f}',
                                        f'{b:.10f}',
                                        hist[i]])
                prev_bin = b

            if count != nbr_act_levels:
                raise ValueError(f'ERROR construct_hist: There are {nbr_act_levels} inputs, but hist contains {count}')


def get_histograms(contribs_path: str, model_structure: list[int], save_path: str) -> None:
    # Getting model structure
    # We consider the last hidden layer
    layer_id = len(model_structure) - 1
    str_layer_id = str(layer_id)

    if len(model_structure) < 1:
        raise ValueError(f'ERROR get_histograms: structure of the model is too short')

    contribs = file_reader(contribs_path, header=p.contribs_header)

    with open(save_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(p.hist_header)

    if len(contribs) == 0:
        return

    # First filtering : filter contrib of given layer
    contribs = [line for line in contribs if (line[p.contrib_layer_id_pos] == str_layer_id)]

    for node_id in range(int(model_structure[layer_id])):
        logger.debug(f'Constructing hist for node {node_id}')
        str_node_id = str(node_id)

        # Third filtering : filter contrib of given node
        contribs_node = [line for line in contribs if line[p.contrib_node_id_pos] == str_node_id]

        # Last filtering : only keep act levels
        contribs_node = [line[p.node_contrib_pos] for line in contribs_node]

        construct_hist(layer_id=str_layer_id,
                       node_id=str_node_id,
                       act_levels=contribs_node,
                       save_path=save_path)
