import csv
from pathlib import Path

import pandas as pd

import config.parameters as p
from config.logger import create_logger
from src.operations.predictor import Predictor, load_model
from src.utils import get_data_loaders, get_target

# Todo

model_path = Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/m4209/_data/model.pt")
db_name = "adult"
data_path = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/m4209/_data/data.csv"
set_name_path = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/m4209/_data/set_name.csv"
save_path = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/m4209/_data/classification.csv"

if __name__ == "__main__":
    logger = create_logger(name=Path(__file__).name, level=p.LOG_LEVEL)

    target = get_target(db_name)

    df = pd.read_csv(data_path, index_col='inputId')
    set_name_column = pd.read_csv(set_name_path, index_col='inputId')
    set_name_column = set_name_column.squeeze()

    loaders = get_data_loaders(df=df, set_name=set_name_column, target=target)

    pred = Predictor(dimensions=load_model(model_path))
    decisions = pred.get_model_decision(loaders['all'])

    _, is_pred_correct, digits = pred.test(loaders['all'])

    content = ['inputId', 'Predicted_class', 'digits_LR', 'digits_HR']
    row_count = 0
    for index, row in df.iterrows():
        predicted_class = not bool(is_pred_correct[row_count]) ^ bool(row[target])
        content.append(
            [index, int(predicted_class), float(digits[row_count][0]), float(digits[row_count][1])])
        row_count += 1

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(content)
