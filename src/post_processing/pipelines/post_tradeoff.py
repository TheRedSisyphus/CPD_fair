from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor, load_model
from src.operations.preprocess import Preprocess
from src.post_processing.FairnessDataset import FairnessDataset, get_performance_criterion
from src.post_processing.pipelines.post_cpl import PostCPL
from src.post_processing.pipelines.roc import RejectOptionClassification
from src.post_processing.pipelines.thresh_op import ThresholdOptimizer
from src.utils import parse_args, get_target, read_parameters, get_data_loaders, get_fairness_criterion_name

if __name__ == '__main__':
    # region PARAMETERS
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL)

    params = read_parameters(args.param, "db_name", "pa_name", "favorable_label", "unfavorable_label",
                             "privileged_group", "unprivileged_group", "fairness_criterion", "performance_criterion",
                             "degradation", "cpl_lb_priv", "cpl_ub_priv", "cpl_lb_unpriv", "cpl_ub_unpriv",
                             "cpl_num_thresh_priv", "cpl_num_thresh_unpriv")

    # endregion

    # region 1. Define preprocessing

    preprocess = Preprocess(db_name=params["db_name"], save_dir=experiment_dir)

    # endregion

    # region 2. Define Fair Dataset
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

    # endregion

    if params["fairness_criterion"] == 'demo_par':
        fair_method = fair_dataset.demo_parity_diff

    elif params["fairness_criterion"] == "eq_opp":
        fair_method = fair_dataset.equal_opportunity_diff

    elif params["fairness_criterion"] == "avg_odds":
        fair_method = fair_dataset.avg_odds_diff

    else:
        raise ValueError

    perf_criterion = get_performance_criterion(params["performance_criterion"])

    # region 3. Apply post CPL

    results_cpd = {"label": "CPD Thresholding",
                   "degradation": [0] + params['degradation'],
                   "fairness": [fair_method("predicted_class")],
                   "performance": [perf_criterion(fair_dataset.df["ground_truth"], fair_dataset.df["predicted_class"])]}

    for d in params['degradation']:
        logger.info(f"== Post-process CPD Thresholding with degradation {d} ==")
        post_cpl = PostCPL(cpl_lb_priv=params["cpl_lb_priv"],
                           cpl_ub_priv=params["cpl_ub_priv"],
                           cpl_lb_unpriv=params["cpl_lb_unpriv"],
                           cpl_ub_unpriv=params["cpl_ub_unpriv"],
                           degrad=d,
                           num_thresh_priv=params["cpl_num_thresh_priv"],
                           num_thresh_unpriv=params["cpl_num_thresh_unpriv"],
                           fairness_criterion=params["fairness_criterion"],
                           performance_criterion=perf_criterion)

        fair_m, perf_m = post_cpl.fit_predict(fair_dataset,
                                              sn_path=Path(experiment_dir) / p.set_name_filename,
                                              save_path=Path(experiment_dir) / 'post_cpl')

        results_cpd["fairness"].append(fair_m)
        results_cpd["performance"].append(perf_m)

    logger.info(results_cpd)

    # endregion

    # 4. Apply ROC

    results_roc = {"label": "ROC",
                   "degradation": [0] + params['degradation'],
                   "fairness": [fair_method("predicted_class")],
                   "performance": [perf_criterion(fair_dataset.df["ground_truth"], fair_dataset.df["predicted_class"])]}

    for d in params['degradation']:
        logger.info(f"== Post-process ROC with degradation {d} ==")

        roc = RejectOptionClassification(num_class_thresh=params['roc_num_class_thresh'],
                                         num_ROC_margin=params['roc_num_ROC_margin'],
                                         low_class_thresh=params['roc_low_class_thresh'],
                                         high_class_thresh=params['roc_high_class_thresh'],
                                         degrad=d,
                                         fairness_criterion=params["fairness_criterion"],
                                         performance_criterion=get_performance_criterion(
                                             params["performance_criterion"]))

        fair_m, perf_m = roc.fit_predict(dataset_pred=fair_dataset, sn_path=Path(experiment_dir) / p.set_name_filename,
                                         save_path=Path(experiment_dir) / 'roc.csv')

        results_roc["fairness"].append(fair_m)
        results_roc["performance"].append(perf_m)

    logger.info(results_roc)

    # endregion

    # 5. Apply Threshop

    results_to = {"label": "TO",
                  "degradation": [0] + params['degradation'],
                  "fairness": [fair_method("predicted_class")],
                  "performance": [perf_criterion(fair_dataset.df["ground_truth"], fair_dataset.df["predicted_class"])]}

    for d in params['degradation']:
        logger.info(f"== Post-process TO with degradation {d} ==")

        thresh_opt = ThresholdOptimizer(scores_lb_priv=params["scores_lb_priv"],
                                        scores_ub_priv=params["scores_ub_priv"],
                                        scores_lb_unpriv=params["scores_lb_unpriv"],
                                        scores_ub_unpriv=params["scores_ub_unpriv"],
                                        degrad=d,
                                        num_thresh_priv=params["threshop_num_thresh_priv"],
                                        num_thresh_unpriv=params["threshop_num_thresh_unpriv"],
                                        fairness_criterion=params["fairness_criterion"],
                                        perf_criterion=get_performance_criterion(params["performance_criterion"]))

        fair_m, perf_m = thresh_opt.fit_predict(dataset_pred=fair_dataset,
                                                sn_path=Path(experiment_dir) / p.set_name_filename,
                                                save_path=Path(experiment_dir) / 'threshop.csv')

        results_to["fairness"].append(fair_m)
        results_to["performance"].append(perf_m)

    logger.info(results_to)


    # endregion

    # 5. region plots
    def plot_fair_degrad(fairness_criterion_name: str, series: list[dict]):
        # remove leading zero
        def remove_z(value, _):
            return f'{value:.3f}'[1:]

        fig, ax = plt.subplots()

        ax.set_xlabel("Degradation allowed")
        ax.xaxis.set_major_formatter(FuncFormatter(remove_z))

        ax.set_ylabel(f"{get_fairness_criterion_name(fairness_criterion_name)} Difference")
        ax.yaxis.set_major_formatter(FuncFormatter(remove_z))

        for i, s in enumerate(series):
            if i == 0:
                marker = 'o'
            elif i == 1:
                marker = 'x'
            else:
                marker = 's'
            ax.plot(s["degradation"], s["fairness"], label=s['label'], marker=marker, markersize=4)
        ax.legend(frameon=False)

        plt.savefig(Path(experiment_dir) / 'tradeoff_fair.pdf', dpi=300)


    def plot_fair_perf(series: list[dict]):
        # remove leading zero
        def remove_z(value, _):
            return f'{value:.3f}'[1:]

        fig, ax = plt.subplots()

        ax.set_xlabel("Degradation allowed")
        ax.xaxis.set_major_formatter(FuncFormatter(remove_z))

        ax.set_ylabel(f"Accuracy Score")
        ax.yaxis.set_major_formatter(FuncFormatter(remove_z))

        for i, s in enumerate(series):
            if i == 0:
                marker = 'o'
            elif i == 1:
                marker = 'x'
            else:
                marker = 's'
            ax.plot(s["degradation"], s["performance"], label=s['label'], marker=marker, markersize=4)
        ax.legend(frameon=False)

        plt.savefig(Path(experiment_dir) / 'tradeoff_perf.pdf', dpi=300)


    plot_fair_degrad(fairness_criterion_name=params["fairness_criterion"],
                     series=[results_cpd, results_roc, results_to])
    plot_fair_perf(series=[results_cpd, results_roc, results_to])

    # endregion

    # Todo : remove logger def from methods
