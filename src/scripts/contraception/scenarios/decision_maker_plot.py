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

#Total women in union using contraceptives
# decision_data = "outputs/run_simulation_nuhdss__2025-05-14T135924.log" #multiplicative
decision_data = "outputs/run_simulation_nuhdss__2025-05-22T154359.log"  #from 2025
log_outcomes = copy.deepcopy(parse_log_file(decision_data, level=logging.DEBUG))
print(log_outcomes['tlo.methods.contraception_nuhdss_slums'].keys())
pop_df = log_outcomes['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary_by_age'].copy()
print(pop_df)
