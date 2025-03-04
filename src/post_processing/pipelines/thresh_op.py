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


class ThresholdOptimizer:
    def __init__(self,
                 perf_criterion: Callable,
                 fairness_criterion: str,
                 num_thresh_priv: int = 100,
                 num_thresh_unpriv: int = 100,
                 scores_lb_priv: float = 0.,
                 scores_ub_priv: float = 1.,
                 scores_lb_unpriv: float = 0.,
                 scores_ub_unpriv: float = 1.,
                 degrad: float = 0.):

        self.num_thresh_priv = num_thresh_priv
        self.num_thresh_unpriv = num_thresh_unpriv

        self.scores_lb_priv = scores_lb_priv
        self.scores_ub_priv = scores_ub_priv
        self.scores_lb_unpriv = scores_lb_unpriv
        self.scores_ub_unpriv = scores_ub_unpriv

        self.degradation = degrad

        self.scores_threshold_priv = None
        self.scores_threshold_unpriv = None

        self.perf_crit = perf_criterion  # What performance criterion is considered
        self.fair_crit = fairness_criterion

    def fit(self, dataset_pred: FairnessDataset):
        logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)
        array_thresh_priv = np.zeros(self.num_thresh_priv * self.num_thresh_unpriv)
        array_thresh_unpriv = np.zeros(self.num_thresh_priv * self.num_thresh_unpriv)

        array_fair_crit = np.zeros(self.num_thresh_priv * self.num_thresh_unpriv)
        array_perf_crit = np.zeros(self.num_thresh_priv * self.num_thresh_unpriv)

        cnt = 0
        for thresh_priv in np.linspace(self.scores_lb_priv, self.scores_ub_priv, self.num_thresh_priv):
            for thresh_unpriv in np.linspace(self.scores_lb_unpriv, self.scores_ub_unpriv, self.num_thresh_unpriv):
                self.scores_threshold_priv = thresh_priv
                self.scores_threshold_unpriv = thresh_unpriv
                array_thresh_priv[cnt] = thresh_priv
                array_thresh_unpriv[cnt] = thresh_unpriv

                dataset_transf_pred = self.predict(dataset_pred)

                if self.fair_crit == 'demo_par':
                    fair_metric = dataset_transf_pred.demo_parity_diff("corrected_class")

                elif self.fair_crit == "eq_opp":
                    fair_metric = dataset_transf_pred.equal_opportunity_diff("corrected_class")

                elif self.fair_crit == "avg_odds":
                    fair_metric = dataset_transf_pred.avg_odds_diff("corrected_class")
                else:
                    raise ValueError(
                        f"Fairness criterion must be either demo_par, eq_opp or avg_odds, got {self.fair_crit}")

                array_fair_crit[cnt] = abs(fair_metric)
                array_perf_crit[cnt] = self.perf_crit(dataset_transf_pred.df["ground_truth"],
                                                      dataset_transf_pred.df["corrected_class"])

                cnt += 1

        # We accept small degradation for fairness
        best_perf_ind = array_perf_crit >= (array_perf_crit.max() - self.degradation)
        logger.debug(f"Best Performance achieved: {array_perf_crit.max()}")
        logger.debug(f"Performance considered: {array_perf_crit[best_perf_ind].tolist()}")

        if any(best_perf_ind):
            best_ind = np.argmin(array_fair_crit[best_perf_ind])
            logger.debug(
                f"Fairness performance considered: {array_fair_crit[best_perf_ind].tolist()}")
            logger.debug(
                f"Performance criterion chosen: {array_perf_crit[best_perf_ind][best_ind]}")
            logger.debug(f"Fairness criterion chosen for priv class: {array_fair_crit[best_perf_ind][best_ind]}")
        else:
            raise ValueError("Unable to satisfy constraints")

        self.scores_threshold_priv = array_thresh_priv[best_perf_ind][best_ind]
        self.scores_threshold_unpriv = array_thresh_unpriv[best_perf_ind][best_ind]

        logger.info(
            f"Found scores threshold for privileged group of {self.scores_threshold_priv}")
        logger.info(
            f"Found scores threshold for unprivileged group of {self.scores_threshold_unpriv}")

        return self

    def predict(self, dataset: FairnessDataset) -> FairnessDataset:
        dataset_new = dataset.df.copy(deep=True)

        # Indices of unprivileged and unfavored to swap
        inds_unpriv = np.logical_and(np.logical_and(dataset.df[dataset.protected_attribute_name] == dataset.unpriv,
                                                    dataset.df["predicted_class"] == dataset.unfav),
                                     dataset.scores_fav > max(self.scores_threshold_unpriv, dataset.scores_fav.min()))
        # Indices of privileged and favored to swap
        inds_priv = np.logical_and(np.logical_and(dataset.df[dataset.protected_attribute_name] == dataset.priv,
                                                  dataset.df["predicted_class"] == dataset.fav),
                                   dataset.scores_fav < min(self.scores_threshold_priv, dataset.scores_fav.max()))

        # Swap individual outcome class
        dataset_new.loc[inds_unpriv, "corrected_class"] = dataset.fav
        dataset_new.loc[inds_priv, "corrected_class"] = dataset.unfav

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
                             "degradation", "scores_lb_priv", "scores_ub_priv", "scores_lb_unpriv", "scores_ub_unpriv",
                             "threshop_num_thresh_priv", "threshop_num_thresh_unpriv")

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

    thresh_opt = ThresholdOptimizer(scores_lb_priv=params["scores_lb_priv"],
                                    scores_ub_priv=params["scores_ub_priv"],
                                    scores_lb_unpriv=params["scores_lb_unpriv"],
                                    scores_ub_unpriv=params["scores_ub_unpriv"],
                                    degrad=params["degradation"],
                                    num_thresh_priv=params["threshop_num_thresh_priv"],
                                    num_thresh_unpriv=params["threshop_num_thresh_unpriv"],
                                    fairness_criterion=params["fairness_criterion"],
                                    perf_criterion=get_performance_criterion(params["performance_criterion"]))

    thresh_opt.fit_predict(dataset_pred=fair_dataset, sn_path=Path(experiment_dir) / p.set_name_filename,
                           save_path=Path(experiment_dir) / 'threshop.csv')

    # endregion
