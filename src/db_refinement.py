from datetime import datetime

import pandas as pd

import config.parameters as p

# Script to transform raw csv to refined csv (before preprocessing)
# Should be run only one time for each db
# This script exists mainly for reproducibility and explainability

if __name__ == "__main__":
    db_name = "diabetes"

    if db_name == "compas":
        data = pd.read_csv(p.db_raw_dir / (db_name + ".csv"))

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

        data.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "adult":  # Todo
        data = pd.read_csv(p.db_raw_dir / db_name)
        data.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')
        raise NotImplementedError

    elif db_name == "credit":
        data = pd.read_excel(p.db_raw_dir / (db_name + ".xls"))
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
        data.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "german_credit":
        data = pd.read_csv(p.db_raw_dir / (db_name + ".csv"))
        data = data.rename(columns={
            "Account Balance": "checking-account",
            "Duration of Credit (month)": "duration",
            "Payment Status of Previous Credit": "credit-history",
            "Purpose": "purpose",
            "Credit Amount": "credit-amount",
            "Value Savings/Stocks": "savings-account",
            "Length of current employment": "employment-since",
            "Instalment per cent": "installment-rate",
            "Sex & Marital Status": "personal-status-and-sex",
            "Guarantors": "other-debtors",
            "Duration in Current address": "residence-since",
            "Most valuable available asset": "property",
            "Age (years)": "age",
            "Concurrent Credits": "other-installment",
            "Type of apartment": "housing",
            "No of Credits at this Bank": "existing-credits",
            "Occupation": "job",
            "No of dependents": "number-people-provide-maintenance-for",
            "Telephone": "telephone",
            "Foreign Worker": "foreign-worker",
            "Creditability": "class-label",
        })
        # Putting target at the end
        target = data.pop("class-label")
        data.insert(len(data.columns), "class-label", target)

        data.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "law":
        from scipy.io import arff

        arff_file = arff.loadarff(p.db_raw_dir / (db_name + ".arff"))
        df = pd.DataFrame(arff_file[0])
        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "oulad":
        df = pd.read_csv(p.db_raw_dir / (db_name + ".csv"))
        df['gender'] = df['gender'].map({"F": 0, "M": 1})
        df['final_result'] = df['final_result'].map({"Fail": 0, "Pass": 1})
        df['disability'] = df['disability'].map({"N": 0, "Y": 1})
        df.drop(columns='id_student', inplace=True)
        df['highest_education'] = df['highest_education'].replace(
            {'No Formal quals': 0., 'Lower Than A Level': 1., 'A Level or Equivalent': 2.,
             'HE Qualification': 3., 'Post Graduate Qualification': 4.})
        df['year'] = df['code_presentation'].str.strip().str[0:4]
        df['semester'] = df['code_presentation'].str.strip().str[-1]
        df.drop(columns="code_presentation", inplace=True)
        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "dutch":
        from scipy.io import arff

        arff_file = arff.loadarff(p.db_raw_dir / (db_name + ".arff"))
        df = pd.DataFrame(arff_file[0])
        df['sex'] = 1 - (df['sex'] - 1).astype(int)
        df['prev_residence_place'] = (df['prev_residence_place'] - 1).astype(int)
        df['occupation'] = 1 - pd.Categorical(df['occupation']).codes
        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "bank":
        df = pd.read_csv(p.db_raw_dir / (db_name + ".csv"), sep=';', quotechar='"')
        df['y'] = df['y'].map({"no": 0, "yes": 1})
        df['loan'] = df['loan'].map({"no": 0, "yes": 1})
        df['housing'] = df['housing'].map({"no": 0, "yes": 1})
        df['default'] = df['default'].map({"no": 0, "yes": 1})
        df = df.sample(frac=1, random_state=99999).reset_index(drop=True)
        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "default":
        df = pd.read_csv(p.db_raw_dir / (db_name + ".csv"))
        df["SEX"] = 1 - df["SEX"]
        df["AGE"] = pd.cut(x=df['AGE'], bins=[min(df['AGE']), 35, 60, max(df['AGE'])], labels=['≤35', '36-60', '>60'],
                           include_lowest=True)

        for c in ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                  'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']:
            assert min(df[c]) < 50001
            assert max(df[c]) > 200000

            df[c] = pd.cut(x=df[c],
                           bins=[min(df[c]), 50001, 200000, max(df[c])],
                           labels=['low', 'medium', 'high'], include_lowest=True)

        for c in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
            assert min(df[c]) <= -1
            assert max(df[c]) >= 3

            df[c] = pd.cut(x=df[c],
                           bins=[min(df[c]), -1, 3, max(df[c])],
                           labels=['pay duly', '1-3 months', '>3 months'], include_lowest=True)

        df.drop(columns=['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], inplace=True)

        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    elif db_name == "diabetes":
        df = pd.read_csv(p.db_raw_dir / (db_name + ".csv"))
        df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'acetohexamide',
                         'tolbutamide', 'troglitazone', 'examide', 'citoglipton', 'glipizide-metformin',
                         'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'], inplace=True)
        df.drop(df[df['race'] == '?'].index, inplace=True)
        df.drop(df[df['diag_1'] == '?'].index, inplace=True)
        df.drop(df[df['diag_2'] == '?'].index, inplace=True)
        df.drop(df[df['diag_3'] == '?'].index, inplace=True)
        df.drop(df[df['readmitted'] == 'NO'].index, inplace=True)
        df['gender'] = df['gender'].map({"Female": 0, "Male": 1})
        df['change'] = df['change'].map({"No": 0, "Ch": 1})
        df['diabetesMed'] = df['diabetesMed'].map({"No": 0, "Yes": 1})
        df['readmitted'] = df['readmitted'].map({">30": 0, "<30": 1})

        # df.drop(columns=['diag_1', 'diag_2', 'diag_3'], inplace=True)
        # Remove not numeric value of diag_1, diag_2 and diag_3
        df = df[pd.to_numeric(df['diag_1'], errors='coerce').notnull()]
        df = df[pd.to_numeric(df['diag_2'], errors='coerce').notnull()]
        df = df[pd.to_numeric(df['diag_3'], errors='coerce').notnull()]

        df.reset_index(drop=True, inplace=True)
        df.to_csv(p.db_refined_dir / (db_name + ".csv"), index=True, index_label='inputId')

    else:
        raise ValueError(f"Invalid db_name : {db_name}")
