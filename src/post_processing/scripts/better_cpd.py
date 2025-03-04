import json

source = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_cpd.json"
dest = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_cpd_better.json"
roc = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_roc.json"

with open(source, 'r') as file:
    data = json.load(file)

with open(roc, 'r') as file:
    roc_data = json.load(file)

content = [["ID", "SENS_ATTR", "ORIGINAL_CLASS", "PREDICTED_CLASS", "CORRECTED_CLASS", "STATUS", "SCORE"]]
for row in data:
    or_class = [line[2] for line in roc_data if line[0] == row[0]]
    if len(or_class) > 1:
        raise ValueError
    or_class = or_class[0]
    pred_class = 'LR' if row[2] in ['neg', 'changedToPos'] else 'HR'
    corr_class = 'LR' if row[2] in ['neg', 'changedToNeg'] else 'HR'
    new_row = [row[0], row[1], or_class, pred_class, corr_class, row[2], row[3]]
    content += [new_row]

content[1:] = sorted(content[1:], key=lambda row_: row_[0])

with open(dest, 'w') as file:
    file.write(json.dumps(content).replace('], [', '],\n['))
