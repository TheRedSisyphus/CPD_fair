import os

import pandas as pd
import seaborn
from matplotlib import pyplot as plt

from config.logger import create_logger
import config.parameters as p

if __name__ == "__main__":
    logger = create_logger(name=os.path.basename(__file__), level=p.LOG_LEVEL)
    base_path = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/datasets/preprocessed/adult_census.csv"
    number_of_corr = 200

    df = pd.read_csv(base_path)
    corr = df.corr()
    plt.figure(figsize=(20, 20))
    seaborn.heatmap(corr, cmap='RdYlGn_r')
    plt.show()


    def get_redundant_pairs(data):
        """Get diagonal and lower triangular pairs of correlation matrix"""
        pairs_to_drop = set()
        cols = data.columns
        for i in range(0, data.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop


    def get_top_abs_correlations(data, n=5):
        au_corr = data.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(data)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]


    pd.set_option('display.max_rows', number_of_corr)
    logger.info(get_top_abs_correlations(df, number_of_corr))
