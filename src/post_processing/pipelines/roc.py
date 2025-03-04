from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor, load_model
from src.operations.preprocess import Preprocess
from src.post_processing.FairnessDataset import FairnessDataset, get_performance_criterion
from src.utils import parse_args, get_target, read_parameters, get_data_loaders


class RejectOptionClassification:
    def __init__(self,
                 num_class_thresh: int,
                 num_ROC_margin: int,
                 low_class_thresh: float,
                 high_class_thresh: float,
                 performance_criterion: Callable,
                 fairness_criterion: str,
                 degrad: float = 0.):
        self.num_class_thresh = num_class_thresh  # Number of thresholds to test
        self.num_ROC_margin = num_ROC_margin  # Number of critical region width to test
        self.low_class_thresh = low_class_thresh  # Lower bound for threshold to test
        self.high_class_thresh = high_class_thresh  # Upper bound for threshold to test
        self.ROC_margin = None
        self.classification_threshold = None
        self.degradation = degrad

        if fairness_criterion not in ['demo_par', 'eq_opp', 'avg_odds', None]:
            raise ValueError("Fairness criterion must be either demo_par, eq_opp or avg_odds")

        if fairness_criterion is None:
            self.fair_crit = 'demo_par'
        else:
            self.fair_crit = fairness_criterion

        self.perf_crit = performance_criterion

    def fit(self, dataset_pred: FairnessDataset):
        logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
        fair_crit_arr = np.zeros(self.num_class_thresh * self.num_ROC_margin)
        perf_crit_arr = np.zeros_like(fair_crit_arr)
        ROC_margin_arr = np.zeros_like(fair_crit_arr)
        class_thresh_arr = np.zeros_like(fair_crit_arr)

        cnt = 0

        # Iterate through class thresholds
        for class_thresh in np.linspace(self.low_class_thresh, self.high_class_thresh, self.num_class_thresh):
            logger.debug(f"Class thresh : {class_thresh}")

            self.classification_threshold = class_thresh
            if class_thresh <= 0.5:
                low_ROC_margin = 0.0
                high_ROC_margin = class_thresh
            else:
                low_ROC_margin = 0.0
                high_ROC_margin = (1.0 - class_thresh)

            # Iterate through ROC margins
            for ROC_margin in np.linspace(low_ROC_margin, high_ROC_margin, self.num_ROC_margin):
                self.ROC_margin = ROC_margin

                # Predict using the current threshold and margin
                dataset_transf_pred = self.predict(dataset_pred)

                ROC_margin_arr[cnt] = self.ROC_margin
                class_thresh_arr[cnt] = self.classification_threshold

                # Balanced accuracy and fairness criterion computations
                if self.fair_crit == 'demo_par':
                    fair_metric = dataset_transf_pred.demo_parity_diff("corrected_class")

                elif self.fair_crit == "eq_opp":
                    fair_metric = dataset_transf_pred.equal_opportunity_diff("corrected_class")

                elif self.fair_crit == "avg_odds":
                    fair_metric = dataset_transf_pred.avg_odds_diff("corrected_class")
                else:
                    raise ValueError(
                        f"Fairness criterion must be either demo_par, eq_opp or avg_odds, got {self.fair_crit}")

                fair_crit_arr[cnt] = abs(fair_metric)
                perf_crit_arr[cnt] = self.perf_crit(dataset_transf_pred.df["ground_truth"],
                                                    dataset_transf_pred.df["corrected_class"])
                cnt += 1

        # We accept small degradation for fairness
        best_perf_ind = perf_crit_arr >= (perf_crit_arr.max() - self.degradation)
        logger.debug(f"Best Performance achieved : {perf_crit_arr.max()}")
        logger.debug(f"Performance considered : {perf_crit_arr[best_perf_ind].tolist()}")

        if any(best_perf_ind):
            best_ind = np.argmin(fair_crit_arr[best_perf_ind])
            logger.debug(f"Fairness performance considered : {fair_crit_arr[best_perf_ind].tolist()}")
            logger.debug(f"Performance criterion chosen : {perf_crit_arr[best_perf_ind][best_ind]}")
            logger.debug(f"Fairness criterion chosen : {fair_crit_arr[best_perf_ind][best_ind]}")
        else:
            raise ValueError("Unable to satisfy constraints")

        self.ROC_margin = ROC_margin_arr[best_perf_ind][best_ind]
        self.classification_threshold = class_thresh_arr[best_perf_ind][best_ind]

        return self

    def predict(self, dataset: FairnessDataset) -> FairnessDataset:
        dataset_new = dataset.df.copy(deep=True)

        fav_pred_inds = (dataset.scores_fav >= min(self.classification_threshold, dataset.scores_fav.max()))
        unfav_pred_inds = ~fav_pred_inds

        y_pred = np.zeros(dataset.scores_fav.shape)
        # Making classification with current self.classification_threshold
        y_pred[fav_pred_inds] = dataset.fav
        y_pred[unfav_pred_inds] = dataset.unfav

        # Indices of critical region around the classification boundary
        crit_region_inds = np.logical_and(
            dataset.scores_fav < min(self.classification_threshold + self.ROC_margin, dataset.scores_fav.max()),  # upper bound
            dataset.scores_fav > max(self.classification_threshold - self.ROC_margin, dataset.scores_fav.min()))  # lower bound

        # Indices of privileged and unprivileged groups
        cond_priv = dataset.df[dataset.protected_attribute_name] == dataset.priv
        cond_unpriv = dataset.df[dataset.protected_attribute_name] == dataset.unpriv

        # New, fairer labels
        dataset_new["corrected_class"] = y_pred

        # Privileged inputs in critical region are assigned to LR
        dataset_new.loc[np.logical_and(crit_region_inds, cond_priv), "corrected_class"] = dataset.unfav

        # Conversely, Unprivileged inputs in critical region are assigned to HR
        dataset_new.loc[np.logical_and(crit_region_inds, cond_unpriv), "corrected_class"] = dataset.fav

        return dataset.create_from(dataset_new)

    def fit_predict(self, dataset_pred: FairnessDataset, sn_path: str | Path, save_path: str | Path, json=False):
        train_data = dataset_pred.get_train_data(set_name_path=sn_path)
        # Fitting on training data, applying on all dataset
        transf = self.fit(train_data).predict(dataset_pred)

        # Parsing to int
        transf.df = transf.df.astype({"ground_truth": int, "predicted_class": int, "corrected_class": int})

        # Export dataset
        transf.df.to_csv(save_path.with_suffix('.csv'), index=False, index_label='inputId')

        if json:
            transf.to_json_array(save_path=save_path)

        # only for tradeoff

        perf_metric = self.perf_crit(transf.df["ground_truth"], transf.df["corrected_class"])
        if self.fair_crit == 'demo_par':
            fair_metric = transf.demo_parity_diff("corrected_class")

        elif self.fair_crit == "eq_opp":
            fair_metric = transf.equal_opportunity_diff("corrected_class")

        elif self.fair_crit == "avg_odds":
            fair_metric = transf.avg_odds_diff("corrected_class")
        else:
            raise ValueError(
                f"Fairness criterion must be either demo_par, eq_opp or avg_odds, got {self.fair_crit}")

        return fair_metric, perf_metric


if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)

    params = read_parameters(args.param, "db_name", "pa_name", "favorable_label", "unfavorable_label",
                             "privileged_group", "unprivileged_group", "fairness_criterion", "performance_criterion",
                             "degradation", "roc_num_class_thresh", "roc_num_ROC_margin", "roc_low_class_thresh",
                             "roc_high_class_thresh")

    # endregion

    # region 1. Define preprocessing

    preprocess = Preprocess(db_name=params["db_name"], save_dir=experiment_dir)

    # endregion

    # region 2. Apply ROC
    predictor = Predictor(dimensions=load_model(experiment_dir.parent / p.data_dir_name / 'model.pt'))
    loaders = get_data_loaders(df=preprocess.df, set_name=preprocess.set_name, target=preprocess.target)

    m_scores = None
    for inputs, _ in loaders['all']:
        m_scores, _ = predictor.forward(inputs)

    if m_scores is None:
        raise ValueError("Model prediction scores are not defined")

    m_scores = torch.exp(m_scores)  # Because logSoftmax is used, we apply exp to have probabilities

    # Not sure about this code
    if params['favorable_label'] == 1 and params['unfavorable_label'] == 0:
        m_scores_class_unfav = pd.Series(m_scores[:, 0].detach().numpy())  # Scores for first class
        m_scores_class_fav = pd.Series(m_scores[:, 1].detach().numpy())  # Scores for second class
    elif params['favorable_label'] == 0 and params['unfavorable_label'] == 1:
        m_scores_class_unfav = pd.Series(m_scores[:, 1].detach().numpy())  # Scores for first class
        m_scores_class_fav = pd.Series(m_scores[:, 0].detach().numpy())  # Scores for second class
    else:
        raise ValueError("Favorable and unfavorable label must be 0 or 1")

    fair_dataset = FairnessDataset(df=Path(experiment_dir) / 'data.csv',
                                   favorable_label=params['favorable_label'],
                                   unfavorable_label=params['unfavorable_label'],
                                   privileged_group=params['privileged_group'],
                                   unprivileged_group=params['unprivileged_group'],
                                   scores_fav=m_scores_class_fav,
                                   scores_unfav=m_scores_class_unfav,
                                   protected_attribute_name=params['pa_name'],
                                   target_name=get_target(params['db_name']),
                                   dir_cpd=experiment_dir.parent / 'cpd')

    roc = RejectOptionClassification(num_class_thresh=params['roc_num_class_thresh'],
                                     num_ROC_margin=params['roc_num_ROC_margin'],
                                     low_class_thresh=params['roc_low_class_thresh'],
                                     high_class_thresh=params['roc_high_class_thresh'],
                                     degrad=params['degradation'],
                                     fairness_criterion=params["fairness_criterion"],
                                     performance_criterion=get_performance_criterion(params["performance_criterion"]))

    roc.fit_predict(dataset_pred=fair_dataset, sn_path=Path(experiment_dir) / p.set_name_filename,
                    save_path=Path(experiment_dir) / 'roc.csv')

    logger.info(f"ROC classification threshold : {roc.classification_threshold}")
    logger.info(f"ROC margin : {roc.ROC_margin}")

    # endregion
