import math
from pathlib import Path

# region PROJECT PATHS
# Project path
project_path = Path("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract")
# experiment file
exp_dir_path = project_path / "experiments"
# experiments for post-processing fairness correction
post_exp_dir_path = project_path / "experiments_post"
# Where original "raw" csv are stored
db_raw_dir = project_path / "raw_data"
# Where original csv post treatment are stored
db_refined_dir = project_path / "refined_csv"
test_db_dir = project_path / "test" / "test_data"
data_dir_name = "_data"
indexes_filename = "indexes.csv"
data_filename = "data.csv"
set_name_filename = "set_name.csv"
protec_attr_filename = "protec_attr.csv"
model_path = "model.pt"
# endregion

# region FILE MANAGEMENT

indexes_header = ['inputId', 'TrueClass', 'PredictedClass', 'SensitiveAttr', 'SetName']
input_id_pos = 0
true_class_pos = 1
pred_class_pos = 2
sens_attr_pos = 3
set_name_pos = 4

contribs_header = ['inputId', 'layerId', 'nodeId', 'nodeContrib']
contrib_layer_id_pos = 1
contrib_node_id_pos = 2
node_contrib_pos = 3

hist_header = ['layerId', 'nodeId', 'binId', 'sigmaInterval_lb', 'sigmaInterval_ub', 'sigmaFreq']
hist_layer_id_pos = 0
hist_node_id_pos = 1
bin_id_pos = 2
sigma_lb_pos = 3
sigma_ub_pos = 4
freq_pos = 5
lh_header = ['inputId', 'Score']
score_pos = 1

# endregion

# region LOGGING
# general level for logging. To specify log_level by file, use LOGLEVEL = None and specify new levels file wise
LOG_LEVEL = "info"
# Logging format : default format display level, date, time and message
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# LOG_FORMAT = "%(message)s"
# endregion

# region CALCULATIONS

# Number of decimal to display (round number after 5th decimal)
DISPLAY_PREC = 4
# Difference between float 'f' and the least value greater than 'f' that is representable as a float.
EPSILON = 1e-10
# Used to specify max number of decimal for float
EPSILON_PREC = int(abs(math.log10(EPSILON)))  # = 10
# Max possible float
POS_UNDEF_FLOAT = 9999.9999999999
# Max possible int
POS_UNDEF_INT = 9999
# Min possible float
NEG_UNDEF_FLOAT = -9999.9999999999
# Min possible int
NEG_UNDEF_INT = -9999
# Used in context when we need finer estimation
LOW_SMOOTHED_PROB = 1e-15
# Used to specify max number of decimal for prob
LSP_PREC = int(abs(math.log10(LOW_SMOOTHED_PROB)))  # = 15

# endregion
