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
decision_data = "outputs/run_simulation_nuhdss__2025-05-15T122915.log"  #from 2025
log_decision = copy.deepcopy(parse_log_file(decision_data, level=logging.DEBUG))
log_decision_numbers = log_decision['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
print(log_decision_numbers)


csv_file_path = outputpath / 'Women_in_union_usingcontraceptives.csv'

    # Save the DataFrame to the CSV file
log_decision_numbers.to_csv(csv_file_path, index=False)

#================== decision makers amomg women in union
log_decision_makers = log_decision['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary_by_decision_maker'].copy()
print(log_decision_makers)

csv_file_path = outputpath / 'Decision_makers_on_contraceptives.csv'

    # Save the DataFrame to the CSV file
log_decision_makers.to_csv(csv_file_path, index=False)







# outcome_baseline = "outputs/run_simulation_nuhdss__2025-05-14T161445.log" #without decision making for women in union
# log_outcomes = copy.deepcopy(parse_log_file(outcome_baseline, level=logging.DEBUG))
# co_sum_df = log_outcomes['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
# print("contraceptives uptake",co_sum_df )
# data_con = pd.DataFrame(co_sum_df)
#      # Define the full path for the CSV file
# csv_file_path = outputpath / 'baseline_daily_contraception_data_union.csv'

#     # Save the DataFrame to the CSV file
# data_con.to_csv(csv_file_path, index=False)


