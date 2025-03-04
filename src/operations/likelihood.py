import csv
import math
import re
from itertools import zip_longest
from pathlib import Path

import config.parameters as p
from config.logger import create_logger

logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)

nk_type = tuple[int, int]
table_type = dict[nk_type, float]
prob_table_type = dict[nk_type, dict[int, float]]


def get_hist_tables(hist: Path) -> tuple[table_type, table_type, prob_table_type]:
    """
    :param hist: Path to where the histogram file is saved
    :return: Three python dictionaries.
    step_table which has keys equal to all node keys of hist file and has values equal to hist bandwidth of the corresponding nodeKey (precision ROUND_PREC)
    lb_table which has keys equal to all node keys of hist file and has values equal to min of contribs of the corresponding nodeKey (precision ROUND_PREC)
    prob_table which has keys equal to all node keys of hist file and has values equal to a python dict {binId: binProbability} (precision ROUND_PREC)
    """
    layerId_pos = 0
    nodeId_pos = 1
    binId_pos = 2
    sigma_interval_lb_pos = 3
    sigma_interval_ub_pos = 4
    sigma_freq_pos = 5

    SIG_EXPECTED_LINE_LEN = 6

    lb_table, step_table, prob_table, hist_table, freq_table = {}, {}, {}, {}, {}

    with open(hist, 'r') as h:
        for no_line, line in enumerate(h):
            line = re.sub('\n$', '', line)
            words = line.split(sep=',')
            if no_line == 0:
                if len(words) != SIG_EXPECTED_LINE_LEN:
                    raise ValueError(
                        f'ERROR get_hist_tables: invalid signature header line length {len(words)} at line {no_line} {line} ({SIG_EXPECTED_LINE_LEN} expected)')

                if words[layerId_pos] != 'layerId':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {layerId_pos} {words[layerId_pos]} (\"layerId\" expected)')

                if words[nodeId_pos] != 'nodeId':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {nodeId_pos} {words[nodeId_pos]} (\"nodeId\" expected)')

                if words[binId_pos] != 'binId':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {binId_pos} {words[binId_pos]} (\"binId\" expected)')

                if words[sigma_interval_lb_pos] != 'sigmaInterval_lb':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {sigma_interval_lb_pos} {words[sigma_interval_lb_pos]} (\"sigmaInterval_lb\" expected)')

                if words[sigma_interval_ub_pos] != 'sigmaInterval_ub':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {sigma_interval_ub_pos} {words[sigma_interval_ub_pos]} (\"sigmaInterval_ub\" expected)')

                if words[sigma_freq_pos] != 'sigmaFreq':
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent column {sigma_freq_pos} {words[sigma_freq_pos]} (\"sigmaFreq\" expected)')

                continue

            if len(words) != SIG_EXPECTED_LINE_LEN:
                raise ValueError(
                    f'ERROR get_hist_tables: invalid signature body line length {len(words)} at line {no_line} {line} ({SIG_EXPECTED_LINE_LEN} expected)')

            try:
                layerId = int(words[layerId_pos])
                nodeId = int(words[nodeId_pos])
            except ValueError:
                raise ValueError(
                    f'ERROR get_hist_tables : line {no_line} of hist file {hist}. Impossible to convert to int one of these : {words[layerId_pos]}, {words[nodeId_pos]}')
            except TypeError:
                raise ValueError(
                    f'ERROR get_hist_tables : line {no_line} of hist file {hist}. Impossible to convert to int one of these : {words[layerId_pos]}, {words[nodeId_pos]}')

            nodeKey = (layerId, nodeId)
            if nodeKey not in lb_table:
                # Counting in the string the number of decimal digits. If higher than 10, raise error, if equal to -1, '.' is not found
                lb_prec = (0 <= words[sigma_interval_lb_pos][::-1].find('.') <= p.EPSILON_PREC)
                if not lb_prec:
                    raise ValueError(
                        f'ERROR get_hist_tables: hist file contains bin edge with a precision higher than {p.EPSILON}, or bin edge that are not float (dot character not found)')

                try:
                    lb_table[nodeKey] = float(words[sigma_interval_lb_pos])
                except ValueError:
                    raise ValueError(
                        f'ERROR get_hist_tables : invalid value to convert to float : {words[sigma_interval_lb_pos]}')

            else:
                if float(words[sigma_interval_lb_pos]) < lb_table[nodeKey]:
                    raise ValueError(
                        f'Lower bound {words[sigma_interval_lb_pos]} at node {nodeKey} is lower than previous lower bound {lb_table[nodeKey]}')

            # Counting in the string the number of decimal digits. If higher than 10, raise error, if equal to -1, '.' is not found
            ub_prec = 0 <= words[sigma_interval_ub_pos][::-1].find('.') <= p.EPSILON_PREC
            lb_prec = 0 <= words[sigma_interval_lb_pos][::-1].find('.') <= p.EPSILON_PREC
            if not ub_prec or not lb_prec:
                raise ValueError(
                    f'ERROR get_hist_tables: hist file contains bin edge with a precision higher than {p.EPSILON}, or bin edge that are not float (dot character not found)')

            try:
                node_step = round(float(words[sigma_interval_ub_pos]) - float(words[sigma_interval_lb_pos]),
                                  p.EPSILON_PREC)
            except ValueError:
                raise ValueError(
                    f'ERROR get_hist_tables : invalid value to convert to float : {words[sigma_interval_ub_pos]} and/or {words[sigma_interval_lb_pos]}')

            if node_step < p.EPSILON:
                logger.warning(f'step for {nodeKey} is null')
                node_step = 0.

            if nodeKey not in step_table:
                step_table[nodeKey] = node_step
            else:
                if abs(step_table[nodeKey] - node_step) > p.EPSILON:
                    raise ValueError(
                        f'ERROR get_hist_tables: inconsistent steps for same node {nodeKey}, and bin Id {words[binId_pos]}. Current step : {node_step} Prev step : {step_table[nodeKey]}')

            try:
                bin_id = int(words[binId_pos])
            except ValueError:
                raise ValueError(f'ERROR get_hist_tables : invalid value to convert to int {words[binId_pos]}')
            try:
                bin_freq = int(words[sigma_freq_pos])
            except ValueError:
                raise ValueError(f'ERROR get_hist_tables : invalid value to convert to int {words[sigma_freq_pos]}')

            if nodeKey not in hist_table:
                hist_table[nodeKey] = {}

            if bin_id in hist_table[nodeKey]:
                raise ValueError(
                    f'ERROR get_hist_tables: invalid duplicate binId {bin_id} on line {no_line} in edge hist table')

            hist_table[nodeKey][bin_id] = bin_freq

            if nodeKey in freq_table:
                freq_table[nodeKey] = freq_table[nodeKey] + bin_freq
            else:
                freq_table[nodeKey] = bin_freq

            for cur_node in hist_table:
                if cur_node not in prob_table:
                    prob_table[cur_node] = {}

                for cur_bin_id in hist_table[cur_node]:
                    if freq_table[cur_node] <= 0:
                        raise ValueError(f'ERROR get_hist_tables: invalid null or negative freq for node {cur_node}')
                    prob_table[cur_node][cur_bin_id] = hist_table[cur_node][cur_bin_id] / freq_table[cur_node]

                    # Rounding prob to precision
                    prob_table[cur_node][cur_bin_id] = round(prob_table[cur_node][cur_bin_id], p.LSP_PREC)

    if len(step_table) == 0 or len(prob_table) == 0 or len(lb_table) == 0:
        logger.warning('Empty hist file')
    return lb_table, step_table, prob_table


def entry_iterator(contribs_path: Path):
    """
    :param contribs_path: The string of the path where the nodes contributions are stored
    :return: Yield a tuple (input_id:str, node_contribution:list[str])
    """
    FILENAME_EXPECTED_LINE_LEN = 4
    with open(contribs_path, 'r') as file:
        try:
            read_lines = file.readlines()
            header = re.sub('\n$', '', read_lines[0])
            file_content = read_lines[1:]
        except (ValueError, IndexError):
            raise ValueError('ERROR entry_iterator: empty contribs file')

        # Header
        if header != 'inputId,layerId,nodeId,nodeContrib':
            raise ValueError(
                f"ERROR entry_iterator: invalid header for contrib file, got '{header}' instead of 'inputId layerId nodeId nodeContrib'")

        contribs = []
        for line_index, (line, next_line) in enumerate(zip_longest(file_content, file_content[1:])):
            line = re.sub('\n$', '', line)
            words = line.strip(' ').split(sep=',')
            if len(words) != FILENAME_EXPECTED_LINE_LEN:
                raise ValueError(
                    f'ERROR entry_iterator: invalid file list line length {len(words)} at line {line_index + 1}, ({FILENAME_EXPECTED_LINE_LEN} expected)')

            contribs.append(words[1:])
            # If line is the last line of the file
            if next_line is None:
                yield words[0], contribs
            # Otherwise, we need to look at the next line of the file to see if the input is different or not
            else:
                next_line = re.sub('\n$', '', next_line)
                next_words = next_line.strip(' ').split(sep=',')

                if words[0] != next_words[0]:
                    yield words[0], contribs
                    contribs = []


def node_profile_iterator(contribs: list[list[str]]):
    """
    :param contribs: List of list of str as returned by entry_iterator
    :return: Yield a tuple (layer_id : int, node_id : int, contrib : Decimal of precision max)
    """
    DATA_layerId_POS = 0
    DATA_nodeId_POS = 1
    DATA_nodeContrib_POS = 2
    DATA_EXPECTED_LINE_LEN = 3

    if len(contribs) == 0:
        raise ValueError('ERROR node_profile_iterator: received empty contribution')

    for line_index, line_contrib in enumerate(contribs):
        if len(line_contrib) != DATA_EXPECTED_LINE_LEN:
            raise ValueError(
                f'ERROR node_profile_iterator: invalid contrib body line length {len(line_contrib)} at line {line_index} ({DATA_EXPECTED_LINE_LEN} expected)')

        try:
            layer_id = int(line_contrib[DATA_layerId_POS])
            node_id = int(line_contrib[DATA_nodeId_POS])
            contrib = float(line_contrib[DATA_nodeContrib_POS])
        except TypeError:
            raise ValueError(
                f'ERROR node_profile_iterator: Impossible to convert ({line_contrib[DATA_layerId_POS]}, {line_contrib[DATA_nodeId_POS]}, {line_contrib[DATA_nodeContrib_POS]}) in (int, int, float)')
        except ValueError:
            raise ValueError(
                f'ERROR node_profile_iterator: Impossible to convert ({line_contrib[DATA_layerId_POS]}, {line_contrib[DATA_nodeId_POS]}, {line_contrib[DATA_nodeContrib_POS]}) in (int, int, float)')

        yield layer_id, node_id, contrib


def get_hist_prob(layerId: int, nodeId: int, node_contrib: float,
                  lb_table: table_type, step_table: table_type, prob_table: prob_table_type) -> float:
    """
    :param layerId: Str of the layer Id
    :param nodeId: Str of the node Id
    :param node_contrib: float of the node contribution
    :param lb_table: Python dict with all nodeKey and minimum contribution as returned by get_hist_tables
    :param step_table: Python dict with all nodeKey and bandwidth of the hist as returned by get_hist_tables
    :param prob_table: Python dict with all nodeKey and nodeProbabilities of the hist as returned by get_hist_tables
    :return: A very low probability if the node doesn't have any contribution in bin, else the bin Probability of the node
    """
    # "Coordonnées" du neurone dans la table parNodeProbTable
    node_key = (layerId, nodeId)

    logger.debug(f'node_key is {node_key}')
    logger.debug(f'node_contrib is {node_contrib:.10f}')

    # node_key not found in histogram file
    if (node_key not in prob_table) or (node_key not in step_table):
        logger.warning(f'missing key in node hist tables {node_key}')
        hist_prob = p.LOW_SMOOTHED_PROB

    # node_key found in histogram file
    else:
        logger.debug(f'node_key {node_key} in hist tables')
        lb_contrib = lb_table[node_key]

        if node_contrib < lb_contrib:
            hist_prob = p.LOW_SMOOTHED_PROB
            logger.debug(f'contrib smaller than min of contributions {lb_contrib:.10f}')

        else:
            if step_table[node_key] > p.EPSILON:
                cur_bin_id = ((node_contrib - lb_contrib) / step_table[node_key])
                cur_bin_id = int(round(cur_bin_id, p.EPSILON_PREC))
                if cur_bin_id < 0:
                    raise ValueError(f'ERROR get_hist_prob : cur_bin_id {cur_bin_id} is undefined')
                logger.debug(f'not null step : curBinId is {cur_bin_id}')

            else:
                if (node_contrib - lb_contrib) > p.EPSILON:
                    cur_bin_id = 1
                    logger.debug(f'null step : contribution is not in hist')
                else:
                    cur_bin_id = 0
                    logger.debug(f'null step : curBinId is {cur_bin_id}')

            if cur_bin_id not in prob_table[node_key]:
                hist_prob = p.LOW_SMOOTHED_PROB
                logger.debug(f'curBinId not in prob_table')

            else:
                hist_prob = prob_table[node_key][cur_bin_id]

    logger.debug(f'hist_prob is {hist_prob:.15f}')

    return hist_prob


def compute_single_class_likelihood(contribs_path: Path, save_path: Path, hist_path: Path):
    """
    :param contribs_path: The string of the path where the nodes contributions are stored
    :param save_path: Path where to save file
    :param hist_path: Path to where the histogram file is saved
    :return: Write in stdout the distance score of each input
    """
    with open(save_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(p.lh_header)

        lb_table, step_table, prob_table = get_hist_tables(hist_path)
        for input_id, contribs in entry_iterator(contribs_path):
            logger.debug(f'input ID is {input_id}')
            current_score = 0.
            for layerId, nodeId, node_contrib in node_profile_iterator(contribs):
                hist_prob = get_hist_prob(layerId, nodeId, node_contrib, lb_table, step_table, prob_table)
                current_score = current_score - math.log(hist_prob)

            logger.debug(f'Score for input {input_id} is {current_score:.10f}\n')
            csvwriter.writerow([input_id, '{0:.{1}f}'.format(current_score, p.DISPLAY_PREC)])
