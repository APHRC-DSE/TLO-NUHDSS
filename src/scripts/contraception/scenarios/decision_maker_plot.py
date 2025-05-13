import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import copy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tlo import Date
from tlo.analysis.utils import extract_results, parse_log_file, summarize

# Define log file paths
# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs





#============================================= decision making ===========================================

decision_data = "outputs/run_simulation_nuhdss__2025-05-13T154827.log"
log_decision = copy.deepcopy(parse_log_file(decision_data, level=logging.DEBUG))
log_decision_numbers = log_decision['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary_by_decision_maker'].copy()
print(log_decision_numbers)


csv_file_path = outputpath / 'decision_making4.csv'

    # Save the DataFrame to the CSV file
log_decision_numbers.to_csv(csv_file_path, index=False)
