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
# Log files (Ensure they exist in the specified path)

log_file_baseline = "outputs/run_analysis_nuhdss__2025-03-27T083217.log"  
# --------------------periodic campaigns

# log_file_campaign_80 = "outputs/run_analysis_nuhdss__2025-03-27T074358.log"  
# log_file_campaign_60 = "outputs/run_analysis_nuhdss__2025-03-27T075826.log"
# log_file_campaign_40 = "outputs/run_analysis_nuhdss__2025-03-27T085730.log"
# log_file_campaign_100 = "outputs/run_analysis_nuhdss__2025-03-27T104457.log"

#semi_annual
# semi_log_file_campaign_40 = "outputs/run_analysis_nuhdss__2025-03-27T090926.log"
# semi_log_file_campaign_60 = "outputs/run_analysis_nuhdss__2025-03-27T105517.log"
# semi_log_file_campaign_80 = "outputs/run_analysis_nuhdss__2025-03-27T110358.log"
# semi_log_file_campaign_100 = "outputs/run_analysis_nuhdss__2025-03-27T111758.log"

#annual
# annual_log_file_campaign_100 = "outputs/run_analysis_nuhdss__2025-03-27T120301.log"
# annual_log_file_campaign_80 = "outputs/run_analysis_nuhdss__2025-03-27T115834.log"
# annual_log_file_campaign_60 = "outputs/run_analysis_nuhdss__2025-03-27T115142.log"
# annual_log_file_campaign_40 = "outputs/run_analysis_nuhdss__2025-03-27T113000.log"

#Biennual
# biennual_log_file_campaign_40 = "outputs/run_analysis_nuhdss__2025-03-27T141439.log"
# biennual_log_file_campaign_60 = "outputs/run_analysis_nuhdss__2025-03-27T143549.log"
# biennual_log_file_campaign_80 = "outputs/run_analysis_nuhdss__2025-03-27T145108.log"
# biennual_log_file_campaign_100 = "outputs/run_analysis_nuhdss__2025-03-27T150617.log"


#-------------------- gradual campaign
#annual
annual_gradual_100 = "outputs/run_analysis_nuhdss__2025-03-27T161217.log"
annual_gradual_80 = "outputs/run_analysis_nuhdss__2025-03-27T162745.log"
annual_gradual_60 = "outputs/run_analysis_nuhdss__2025-03-27T163420.log"

#semmiannual
# semiannual_gradual_100 = "outputs/run_analysis_nuhdss__2025-03-27T165525.log"
# semiannual_gradual_80 = "outputs/run_analysis_nuhdss__2025-03-27T165106.log"
# semiannual_gradual_60 = "outputs/run_analysis_nuhdss__2025-03-27T164628.log"

#Quartely
# quartely_gradual_100 = "outputs/run_analysis_nuhdss__2025-03-28T093056.log"
# quartely_gradual_80 = "outputs/run_analysis_nuhdss__2025-03-28T092717.log"
# quartely_gradual_60 = "outputs/run_analysis_nuhdss__2025-03-28T091955.log"

#Biennual
# biennial_gradual_100 = "outputs/run_analysis_nuhdss__2025-03-28T093708.log"
# biennial_gradual_80 = "outputs/run_analysis_nuhdss__2025-03-28T094452.log"
# biennial_gradual_60 = "outputs/run_analysis_nuhdss__2025-03-28T095048.log"

#Static
# static_100 = "outputs/run_analysis_nuhdss__2025-03-28T152324.log"
# static_80 = "outputs/run_analysis_nuhdss__2025-03-28T151958.log"
# static_60 = "outputs/run_analysis_nuhdss__2025-03-28T151609.log"
# static_40 = "outputs/run_analysis_nuhdss__2025-03-28T151047.log"

#-------------------- different cmpaigns at 60% and annual
# static = "outputs/run_analysis_nuhdss__2025-03-28T151609.log"
# gradual = "outputs/run_analysis_nuhdss__2025-03-27T163420.log"
# periodic = "outputs/run_analysis_nuhdss__2025-03-27T115142.log"


# Parse log files
log_baseline = copy.deepcopy(parse_log_file(log_file_baseline, level=logging.DEBUG))
log_campaign_100 = copy.deepcopy(parse_log_file(annual_gradual_100, level=logging.DEBUG))
log_campaign_80 = copy.deepcopy(parse_log_file(annual_gradual_80, level=logging.DEBUG))
log_campaign_60 = copy.deepcopy(parse_log_file(annual_gradual_60, level=logging.DEBUG))
#log_campaign_40 = copy.deepcopy(parse_log_file(static_40, level=logging.DEBUG))



contraception_baseline = log_baseline['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
contraception_campaign_60 = log_campaign_60['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
contraception_campaign_80 = log_campaign_80['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
#contraception_campaign_40 = log_campaign_40['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
contraception_campaign_100 = log_campaign_100['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()

# Function to process the contraception data
def process_contraception_data(df, label):
    df["total_women_daily"] = df[["IUD", "implant", "injections", "male_condom", "not_using",
                                  "other_modern", "other_traditional", "pill", "rhythm"]].sum(axis=1)
    df["total_using_modern"] = df[["IUD", "implant", "injections", "male_condom", "other_modern", "pill"]].sum(axis=1)
    df["daily_prevalence"] = (df["total_using_modern"] / df["total_women_daily"]) * 100
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Calculate yearly MCPR
    yearly_prevalence = df.groupby("year")["daily_prevalence"].mean().reset_index(name="MCPR")

    # Extract last recorded total_women_daily per year
    yearly_total = df.loc[df.groupby("year")["date"].idxmax(), ["year", "total_women_daily"]]

    # Merge
    model_data = pd.merge(yearly_total, yearly_prevalence, on="year")

    # Calculate Confidence Interval
    model_data["mcpr"] = model_data["MCPR"] / 100
    model_data["se"] = np.sqrt(model_data["mcpr"] * (1 - model_data["mcpr"]) / model_data["total_women_daily"])
    model_data["ci_lower"] = np.maximum(0, model_data["mcpr"] - 1.96 * model_data["se"]) *100
    model_data["ci_upper"] = np.minimum(1, model_data["mcpr"] + 1.96 * model_data["se"]) *100
    model_data["mcpr"] *= 100 

    model_data["Scenario"] = label
    return model_data

# Process data
processed_baseline = process_contraception_data(contraception_baseline, "Baseline")
processed_campaign_80 = process_contraception_data(contraception_campaign_80, "Campaign 80%")
processed_campaign_60 = process_contraception_data(contraception_campaign_60, "Campaign 60%")
#processed_campaign_40 = process_contraception_data(contraception_campaign_40, "Campaign 40%")
processed_campaign_100 = process_contraception_data(contraception_campaign_100, "Campaign 100%")
# Combine both datasets
#


# print(processed_baseline[['year', 'mcpr']])
# print(processed_campaign_80[['year', 'mcpr']])

#combined_mcpr = pd.concat([processed_baseline, processed_campaign_40])

combined_mcpr = pd.concat([processed_baseline,processed_campaign_80,processed_campaign_60,processed_campaign_100])
#print(combined_mcpr)
#print(combined_mcpr["Scenario"].unique())


# Plot MCPR for both scenarios
plt.figure(figsize=(9, 5))
sns.lineplot(data=combined_mcpr, x="year", y="mcpr", hue="Scenario", marker="o", linewidth=1, alpha=0.7)

plt.fill_between(processed_baseline["year"], processed_baseline["ci_lower"], processed_baseline["ci_upper"], alpha=0.2, color="blue")
plt.fill_between(processed_campaign_80["year"], processed_campaign_80["ci_lower"], processed_campaign_80["ci_upper"], alpha=0.2, color="orange")
plt.fill_between(processed_campaign_60["year"], processed_campaign_60["ci_lower"], processed_campaign_60["ci_upper"], alpha=0.2, color="green")
#plt.fill_between(processed_campaign_40["year"], processed_campaign_40["ci_lower"], processed_campaign_40["ci_upper"], alpha=0.2, color="teal")
plt.fill_between(processed_campaign_100["year"], processed_campaign_100["ci_lower"], processed_campaign_100["ci_upper"], alpha=0.2, color="purple")

# Bold title and labels
plt.title("MCPR Over Time for Campaigns", fontsize=14, fontweight="bold")
plt.xlabel("Year", fontsize=12, fontweight="bold")
plt.ylabel("MCPR (%)", fontsize=12, fontweight="bold")

plt.ylim(0, 100)
plt.xticks(np.arange(2010, 2041, 3), fontsize=10, fontweight="bold")
plt.yticks(np.arange(0, 100, 10), fontsize=10, fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Scenario", title_fontsize=12, fontsize=10)

plt.show()

#--------------------------------------- yearly data ------------------------------------------------
#----------------------------------------- population data ---------------------------------- 
#----------------------------------------- population data ---------------------------------- 
#----------------------------------------- population data ---------------------------------- 
outcome_baseline = "outputs/run_analysis_nuhdss__2025-04-03T104941.log"
log_outcomes = copy.deepcopy(parse_log_file(outcome_baseline, level=logging.DEBUG))
pop_df = log_outcomes['tlo.methods.contraception_nuhdss_slums']['sex_distribution_summary'].copy()

pop_df['date'] = pd.to_datetime(pop_df['date'])
    # Extract year from the date column
pop_df['year'] = pop_df['date'].dt.year
    # Exclude datetime columns
    
    # Get last entry for each year
yearly_data = pop_df.sort_values('date').groupby('year').last().reset_index()

    # Compute total population
yearly_data['Total'] = yearly_data['F'] + yearly_data['M']

    # Print yearly population data
    #print("Yearly Population Data:\n", yearly_data[['year', 'F', 'M', 'Total']])
csv_file_path = outputpath / 'yearly_data.csv'

    # Save the DataFrame to the CSV file
yearly_data.to_csv(csv_file_path, index=False)
 
#----------------------------------------- outcomes of pregancies --------------------------------------

outcome_baseline = "outputs/run_analysis_nuhdss__2025-04-03T104941.log"
log_outcomes = copy.deepcopy(parse_log_file(outcome_baseline, level=logging.DEBUG))
pregnacy_outcome = log_outcomes['tlo.methods.contraception_nuhdss_slums']['pregnancy_outcome'].copy()
    # Convert 'date' column to datetime 
pregnacy_outcome['date'] = pd.to_datetime(pregnacy_outcome['date'])

    # Extract the year into a new column
pregnacy_outcome['year'] = pregnacy_outcome['date'].dt.year
outcome_counts = pregnacy_outcome.groupby(['year', 'outcomes']).size().reset_index(name='count')
outcome_counts = outcome_counts.pivot(index='year', columns='outcomes', values='count')
outcome_counts = outcome_counts.reset_index()
print("Pregnancy outcome data", outcome_counts )

      # Define the full path for the CSV file
csv_file_path = outputpath / 'yearly_pregnacy_outcomes_baseline.csv'

    # Save the DataFrame to the CSV file
outcome_counts.to_csv(csv_file_path, index=False)


################### outcomes with gradual campaign at cmax= 60%

outcomes_campaign = "outputs/run_analysis_nuhdss__2025-04-03T114022.log"
log_outcomes = copy.deepcopy(parse_log_file(outcomes_campaign, level=logging.DEBUG))
pregnacy_outcome = log_outcomes['tlo.methods.contraception_nuhdss_slums']['pregnancy_outcome'].copy()
    # Convert 'date' column to datetime 
pregnacy_outcome['date'] = pd.to_datetime(pregnacy_outcome['date'])

    # Extract the year into a new column
pregnacy_outcome['year'] = pregnacy_outcome['date'].dt.year
outcome_counts = pregnacy_outcome.groupby(['year', 'outcomes']).size().reset_index(name='count')
outcome_counts = outcome_counts.pivot(index='year', columns='outcomes', values='count')
outcome_counts = outcome_counts.reset_index()
print("Pregnancy outcome data", outcome_counts )

      # Define the full path for the CSV file
csv_file_path = outputpath / 'yearly_pregnacy_outcomes_campaign.csv'

    # Save the DataFrame to the CSV file
outcome_counts.to_csv(csv_file_path, index=False)











