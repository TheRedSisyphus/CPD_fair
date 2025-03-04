from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor, load_model
from src.operations.preprocess import Preprocess
from src.utils import parse_args, get_target, read_parameters, get_map_oc_str, get_map_sc_str, get_data_loaders, \
    df_to_json_array, FairnessDataset


class ThresholdOptimizer:
    def __init__(self,
                 target_ratio: float,
                 metric: Callable,
                 low_scores_thresh: float = 0.0,
                 high_scores_thresh: float = 0.99,
                 num_scores_thresh: int = 100,
                 density_class_thresh: int = 100,
                 seed: int = 999999):

        if low_scores_thresh is None:
            low_scores_thresh = 0.
        self.low_scores_thresh = low_scores_thresh  # Lower Bound for threshold, default is 0
        if high_scores_thresh is None:
            high_scores_thresh = 0.99
        self.high_scores_thresh = high_scores_thresh  # Upper Bound for threshold, default is 0.99
        if num_scores_thresh is None:
            num_scores_thresh = 100
        self.num_scores_thresh = num_scores_thresh  # Number of threshold tested between lower and upper bound
        if density_class_thresh is None:
            density_class_thresh = 100
        self.density_class_thresh = density_class_thresh  # Number of threshold of switching class tested density_class_thresh
        if seed is None:
            seed = 999999
        self.seed = seed  # For reproducibility
        self.target_ratio = target_ratio  # Ratio we want to achieve. If set to 0.75 for instance, then the method tries to reach P(LR|F) = P(LR|M) = 0.75
        self.metric = metric  # What performance metric is considered

    def fit(self,
            dataset_pred: FairnessDataset,
            desired_classes: list[bool | int | float],
            save_path: str | Path,
            map_oc: dict[int, str] = None,
            map_sc: dict[int, str] = None):
        np.random.seed(self.seed)
        predictions_classes = {}
        ground_truth_column = dataset_pred.df[dataset_pred.target_name]
        for desired_class in desired_classes:
            # region Find classification threshold for which target ratio is reached
            ratio_table = {}

            for scores_thresh in np.linspace(self.low_scores_thresh, self.high_scores_thresh, self.num_scores_thresh):
                ratio = sum(
                    np.logical_and(dataset_pred.scores < scores_thresh,
                                   dataset_pred.protected_attribute == desired_class)) / sum(
                    dataset_pred.protected_attribute == desired_class)
                ratio_table[scores_thresh] = ratio

            closest_hard_thresh = min(ratio_table.keys(), key=lambda x: abs(ratio_table[x] - self.target_ratio))
            gap = abs(ratio_table[closest_hard_thresh] - self.target_ratio)
            logger.info(f"Closest ratio found with gap of {gap}")
            if gap <= 1 / dataset_pred.df[dataset_pred.protected_attribute == desired_class].shape[0]:
                logger.info(f"Maximum precision reached !")
            else:
                logger.info(
                    f"Please increase num_scores_thresh arguments to reach max precision of {1 / dataset_pred.df[dataset_pred.protected_attribute == desired_class].shape[0]}")
            # endregion

            # Compute probabilities of switching class with density density_class_thresh
            prob_table = {}
            for thresh in np.linspace(closest_hard_thresh, self.high_scores_thresh, self.density_class_thresh):
                scores_for_class = dataset_pred.scores[
                    dataset_pred.df[dataset_pred.protected_attribute_name] == desired_class]
                prob_table[thresh] = 1 - ((scores_for_class <= closest_hard_thresh).values.sum() / (
                        scores_for_class <= thresh).values.sum())

            # Compute metrics
            metrics_table = {}
            for thresh, prob in prob_table.items():
                hard_pred = dataset_pred.scores[
                                dataset_pred.df[dataset_pred.protected_attribute == desired_class].index] >= thresh
                # Portion of dataframe of desired class that are not selected by hard threshold
                desi_class_not_fav = dataset_pred.df[
                    (dataset_pred.protected_attribute == desired_class) & (dataset_pred.scores < thresh)]
                nbr_to_select = int(len(desi_class_not_fav) * prob)
                soft_select = np.random.choice(hard_pred.index, size=nbr_to_select, replace=False)
                soft_pred = pd.Series(data=0, index=hard_pred.index)
                soft_pred[soft_select] = 1

                y_true = dataset_pred.df[dataset_pred.protected_attribute == desired_class][dataset_pred.target_name]
                y_pred = np.logical_or(hard_pred, soft_pred)
                try:
                    metric = self.metric(y_true, y_pred)
                except (TypeError, ValueError):
                    raise ValueError("Metric function is not correct. "
                                     "The function must take y_pred and y_true as arguments "
                                     "and return a numerical value")

                metrics_table[thresh] = {"metric": metric,
                                         "soft_prediction": soft_pred,
                                         "hard_prediction": hard_pred,
                                         "prob": prob}

            # Best threshold is the threshold that maximize performance metrics chosen
            best_thresh = max(metrics_table, key=lambda x: metrics_table.get(x).get("metric"))

            # Log threshold optimizer infos
            logger.info(f"Desired class : {map_sc[desired_class]}")
            logger.info(f"Hard Threshold {best_thresh}")
            logger.info(f"Probability for soft threshold : {metrics_table[best_thresh]["prob"]}")
            logger.info(f"Metric {self.metric.__name__} : {metrics_table[best_thresh]["metric"]}")

            # Store new labels in predictions_classes
            predictions_classes[desired_class] = np.logical_or(metrics_table[best_thresh]["soft_prediction"],
                                                               metrics_table[best_thresh]["hard_prediction"])

        # Apply the corrections to the dataset
        dataset_pred.df[dataset_pred.target_name] = np.full(dataset_pred.scores.shape[0], False)
        for c, p_c in predictions_classes.items():
            dataset_pred.df.loc[dataset_pred.protected_attribute == c, dataset_pred.target_name] = np.logical_or(
                dataset_pred.df.loc[dataset_pred.protected_attribute == c, dataset_pred.target_name], p_c)

        # export dataset
        dataset_pred.df[dataset_pred.target_name] = dataset_pred.df[dataset_pred.target_name].astype(int)
        dataset_pred.df["ground_truth"] = ground_truth_column
        dataset_pred.df.rename(columns={dataset_pred.target_name: "corrected_class"},
                               inplace=True)
        dataset_pred.df["predicted_class"] = dataset_pred.scores.round().astype(int)

        oc_0, oc_1 = list(map_oc.values())
        sc_0, sc_1 = list(map_sc.values())
        df_to_json_array(fairness_df=dataset_pred,
                         save_path=save_path,
                         class_0_name=oc_0,
                         class_1_name=oc_1,
                         files_cpd_class_0=[experiment_dir.parent / 'cpd' / f'lh_{sc_0}-{oc_0}.csv',
                                            experiment_dir.parent / 'cpd' / f'lh_{sc_1}-{oc_0}.csv'],
                         files_cpd_class_1=[experiment_dir.parent / 'cpd' / f'lh_{sc_0}-{oc_1}.csv',
                                            experiment_dir.parent / 'cpd' / f'lh_{sc_1}-{oc_1}.csv'],
                         map_oc=map_oc,
                         map_sc=map_sc)


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)

    params = read_parameters(args.param, "db_name", "pa_name", "favorable_label", "unfavorable_label",
                             "thresh_low_scores", "thresh_high_scores", "thresh_num_scores", "thresh_density_class",
                             "thresh_seed", "thresh_target_ratio", "thresh_desired_classes")

    # endregion

    # region 1. Define preprocessing

    preprocess = Preprocess(db_name=params["db_name"], save_dir=experiment_dir)

    # endregion

    # region 2. Apply Threshold Optimizer

    predictor = Predictor(dimensions=load_model(experiment_dir.parent / p.data_dir_name / 'model.pt'))
    loaders = get_data_loaders(df=preprocess.df, set_name=preprocess.set_name, target=preprocess.target)

    m_scores = None
    for inputs, _ in loaders['all']:
        m_scores, _ = predictor.forward(inputs)

    if m_scores is None:
        raise ValueError("Model prediction scores are not defined")

    m_scores = torch.exp(m_scores)  # Because logSoftmax is used, we apply exp to have probabilities
    m_scores = m_scores[:, 1].detach().numpy()  # Scores for second class
    m_scores = pd.Series(m_scores)

    adult = FairnessDataset(path=Path(experiment_dir) / 'data.csv',
                            favorable_label=params['favorable_label'],
                            unfavorable_label=params['unfavorable_label'],
                            scores=m_scores,
                            protected_attribute_name=params['pa_name'],
                            target_name=get_target(params['db_name']))

    thresh_opt = ThresholdOptimizer(target_ratio=params["thresh_target_ratio"],
                                    metric=precision_score,
                                    low_scores_thresh=params["thresh_low_scores"],
                                    high_scores_thresh=params["thresh_high_scores"],
                                    num_scores_thresh=params["thresh_num_scores"],
                                    density_class_thresh=params["thresh_density_class"],
                                    seed=params["thresh_seed"])

    thresh_opt.fit(adult,
                   desired_classes=params["thresh_desired_classes"],
                   save_path=Path(experiment_dir) / 'threshop.json',
                   map_oc=get_map_oc_str(get_target(params["db_name"])),
                   map_sc=get_map_sc_str(params["pa_name"]))

    # endregion
