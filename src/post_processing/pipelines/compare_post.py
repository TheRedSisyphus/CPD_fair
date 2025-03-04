import itertools
from pathlib import Path

import pandas as pd

import config.parameters as p
from config.logger import create_logger
from src.post_processing.FairnessDataset import FairnessDataset
from src.utils import parse_args, read_parameters, get_target

if __name__ == "__main__":
    args = parse_args()
    experiment_dir = Path(args.param).parent
    logger = create_logger(name=Path(__file__).stem, level=p.LOG_LEVEL, log_format="%(message)s")
    params = read_parameters(args.param, "db_name", "pa_name", "favorable_label", "unfavorable_label",
                             "privileged_group", "unprivileged_group", )

    content_to_compare = {}
    for root, _, list_of_files in experiment_dir.walk():
        for files in list_of_files:
            if files in ['roc.csv', 'threshop.csv', 'post_cpl.csv']:
                content_to_compare[Path(root / files).stem] = FairnessDataset(
                    df=pd.read_csv(root / files, index_col='inputId'),
                    favorable_label=params["favorable_label"],
                    unfavorable_label=params["unfavorable_label"],
                    privileged_group=params["privileged_group"],
                    unprivileged_group=params["unprivileged_group"],
                    protected_attribute_name=params["pa_name"],
                    target_name=get_target(params["db_name"]),
                    dir_cpd=root.parent/'cpd')

    logger.info(f"Detected files : {[f for f in content_to_compare.keys()]}")

    # region display model precision before correction

    first_content = content_to_compare[next(iter(content_to_compare))]
    model_prec_str = (f"Model accuracy before correction is:\n"
                      f"all: {first_content.df[first_content.df["ground_truth"] == first_content.df["predicted_class"]].shape[0] * 100 / first_content.df.shape[0]:.2f} %")
    model_prec_str += "".join([
        f"\n{first_content.map_sc[pa]}: {(first_content.df[(first_content.df["ground_truth"] == first_content.df["predicted_class"]) & (first_content.df[first_content.protected_attribute_name] == pa)].shape[0]) * 100 / first_content.df[first_content.df[first_content.protected_attribute_name] == pa].shape[0]:.2f} %"
        for pa in [params["privileged_group"], params["unprivileged_group"]]])
    logger.info(model_prec_str + "\n")

    # endregion

    # region display fairness metrics before correction
    logger.info("\nFairness metric before correction :\n"
                f"Demographic Parity difference : {first_content.demo_parity_diff("predicted_class")}\n"
                f"Equal Opportunity difference : {first_content.equal_opportunity_diff("predicted_class")}\n"
                f"Average Odds difference : {first_content.avg_odds_diff("predicted_class")}\n"
                "\n")

    # endregion

    for name, content_to_study in content_to_compare.items():
        logger.info(f"==================== Results for method {name} ====================")
        description = content_to_study.describe()
        logger.info(description)

    for pair_of_contents in itertools.combinations(content_to_compare, 2):
        logger.info(
            f"==================== Comparison of {pair_of_contents[0]} and {pair_of_contents[1]} ====================")
        frst_method_name, scd_method_name = pair_of_contents
        comparison = content_to_compare[frst_method_name].compare(other=content_to_compare[scd_method_name],
                                                                  method_name=frst_method_name,
                                                                  other_method_name=scd_method_name)
        logger.info(comparison)
