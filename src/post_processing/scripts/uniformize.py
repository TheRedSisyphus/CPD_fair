import json

roc = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_roc.json"
threshop = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_threshop.json"
cpd = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results_cpd_better.json"
destination = "/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/experiments_post/results.json"

with open(roc, 'r') as file:
    data_roc = json.load(file)

with open(threshop, 'r') as file:
    data_thresh = json.load(file)

with open(cpd, 'r') as file:
    data_cpd = json.load(file)

content = [
    ["ID", "SENS_ATTR", "ORIGINAL_CLASS", "PREDICTED_CLASS", "CORRECTED_CLASS", "STATUS", "MODEL_SCORE", "CPD_SCORE"]]

for row in data_cpd[1:]:
    # InputId, format int
    inputId = row[0]
    # Sensitive attribute, format str
    sens_attr = row[1]
    # Ground truth output class, format str
    or_class = row[2]
    # Output class as predicted by model, format str
    pred_class = row[3]
    # Output class after post-processing correction, format str
    corr_class = row[4]
    # "pos", "neg" (as pred by the model)
    # "changedToPos", "changedToNeg" if post-processing correction changes model decision, format str
    status = row[5]
    # Score returned by the model, format float
    model_score = [line[6] for line in data_roc if line[0] == row[0]]
    # Normalized CPD score, format float
    cpd_score = row[6]
    assert len(model_score) == 1
    model_score = model_score[0]
    new_row = [inputId, sens_attr, or_class, pred_class, corr_class, status, model_score, cpd_score]
    content += [new_row]

# Sort content by id
content[1:] = sorted(content[1:], key=lambda row_: row_[0])

with open(destination, 'w') as file:
    file.write(json.dumps(content).replace('], [', '],\n['))
