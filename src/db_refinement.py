import os.path
from datetime import datetime

import pandas as pd

import config.parameters as p

# Script to transform raw csv to refined csv (before preprocessing)
# Should be run only one time for each db
# This script exists mainly for reproducibility and explainability

if __name__ == "__main__":
    db_name = "credit"

    if db_name == "compas":
        data = pd.read_csv(os.path.join(p.db_raw_dir, db_name + ".csv"))

        data = data[(data["days_b_screening_arrest"] <= 30)
                    & (data["days_b_screening_arrest"] >= -30)
                    & (data["is_recid"] != -1)
                    & (data["c_charge_degree"] != 'O')
                    & (data["score_text"] != 'N/A')].reset_index(drop=True)

        date_format = '%Y-%m-%d %I:%M:%S'
        data['c_jail_out'] = data['c_jail_out'].apply(lambda x: datetime.strptime(x, date_format))
        data['c_jail_in'] = data['c_jail_in'].apply(lambda x: datetime.strptime(x, date_format))
        data['length_of_stay'] = (data['c_jail_out'] - data['c_jail_in']).apply(
            lambda x: int(abs(x.total_seconds())))
        data = data[['sex', 'age', 'race',
                     'juv_fel_count',  # number of juvenile felonies
                     'juv_misd_count',  # number of juvenile misdemeanors
                     'length_of_stay',
                     # number of prior juvenile convictions
                     # that are not considered either felonies or misdemeanors
                     'juv_other_count',
                     'priors_count',  # Nombre d'antécédents
                     'decile_score',  # Risk of recidivism
                     'c_charge_degree',  # The charge degree of defendants. F: Felony M: Misdemeanor
                     'two_year_recid'  # Target
                     ]]

        data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
        data['c_charge_degree'] = data['c_charge_degree'].map({'F': 1, 'M': 0})

        data.to_csv(os.path.join(p.db_refined_dir, db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "adult":  # Todo
        data = pd.read_csv(os.path.join(p.db_raw_dir, db_name))
        data.to_csv(os.path.join(p.db_refined_dir, db_name + ".csv"), index=True, index_label='inputId')
        raise NotImplementedError

    elif db_name == "credit":
        data = pd.read_excel(os.path.join(p.db_raw_dir, db_name + ".xls"))
        # Remove ID column
        data.drop(columns=['Unnamed: 0'], inplace=True)
        # First row is the header
        data.columns = data.iloc[0]
        data = data[1:]
        data.index -= 1

        # Lower all column names
        data.columns = [x.lower() for x in data.columns]
        #  0 for Female and 1 for Male
        data['sex'] = data['sex'].map({2: 0, 1: 1})
        data.to_csv(os.path.join(p.db_refined_dir, db_name + ".csv"), index=True, index_label='inputId')

    else:
        raise ValueError(f"Invalid db_name : {db_name}")
