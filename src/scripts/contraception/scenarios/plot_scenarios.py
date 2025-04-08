import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tlo import Date
from tlo.analysis.utils import extract_results, parse_log_file, summarize

# Define log file paths
without_campaign = "outputs//Periodic_Campaign__2025-03-26T233141.log"
with_campaign = "outputs//Periodic_Campaign__2025-03-26T233244.log"

# Parse log files
log_df_without = parse_log_file(without_campaign, level=logging.DEBUG)
log_df_with = parse_log_file(with_campaign, level=logging.DEBUG)

# Print parsed log structure
print("Log dataframe without campaign:\n", log_df_without)
print("Log dataframe with campaign:\n", log_df_with)

# Extract contraception use summary if available
if 'tlo.methods.contraception_nuhdss_slums' in log_df_with:
    print("Keys available in contraception log:", log_df_with['tlo.methods.contraception_nuhdss_slums'].keys())
    
    # Extract summary
    co_sum_df_without = log_df_without['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
    co_sum_df_with = log_df_with['tlo.methods.contraception_nuhdss_slums']['contraception_use_summary'].copy()
    
    print("Contraceptive uptake without campaign:\n", co_sum_df_without)
    print("Contraceptive uptake with campaign:\n", co_sum_df_with)

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
        model_data["ci_lower"] = np.maximum(0, model_data["mcpr"] - 1.96 * model_data["se"])
        model_data["ci_upper"] = np.minimum(1, model_data["mcpr"] + 1.96 * model_data["se"])

        model_data["Scenario"] = label
        return model_data

    # Process extracted data
    mcpr_without = process_contraception_data(co_sum_df_without, "Without Campaign")
    mcpr_with = process_contraception_data(co_sum_df_with, "With Campaign")

    # Combine both datasets
    combined_mcpr = pd.concat([mcpr_without, mcpr_with])

    # Plot MCPR for both scenarios
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=combined_mcpr, x="year", y="mcpr", hue="Scenario", marker="o", linewidth=1.2)
    plt.fill_between(mcpr_without["year"], mcpr_without["ci_lower"], mcpr_without["ci_upper"], alpha=0.2, color="blue")
    plt.fill_between(mcpr_with["year"], mcpr_with["ci_lower"], mcpr_with["ci_upper"], alpha=0.2, color="orange")

    plt.title("MCPR Over Time")
    plt.xlabel("Year")
    plt.ylabel("MCPR (%)")
    plt.ylim(0, 0.7)
    plt.xticks(np.arange(2010, 2041, 3))
    plt.yticks(np.arange(0, 0.71, 0.05))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Scenario")

    plt.show()

else:
    print("Key 'tlo.methods.contraception_nuhdss_slums' not found in log files.")
    