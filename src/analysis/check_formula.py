# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: check_forumal.py - tests formula on csv file to validate results.
# Mark Zhdan | 01-05-2023

import pandas as pd


def calculate_vlr_rating(*stats):
    # Coefficients will be updated after running analyze_data.py
    # Placeholder for now; run analyze_data.py to get updated values
    coefficients = [1.041081241376, 0.267495815831, -1.090621026753, 0.002527464479, -0.070160070478]
    intercept = 0.9277842347983403

    rating = sum(coef * stat for coef, stat in zip(coefficients, stats)) + intercept
    return rating


df = pd.read_csv("data/clean/stats_recent.csv")

featureKeys = ["KPR", "APR", "DPR", "ADRa", "KAST"]

player_name_width = max(len(row["Player"]) for index, row in df.iterrows()) + 2

# Iterate through each row and calculate the rating
for index, row in df.iterrows():
    # Extract the stats
    stats = row[featureKeys]

    # Compare with actual rating
    actual_rating = row["R2.0"]
    predicted_rating = calculate_vlr_rating(*stats)
    error = actual_rating - predicted_rating

    player_name = row["Player"].ljust(player_name_width)
    print(
        f"{player_name} | Actual: {actual_rating:.2f}, Predicted: {predicted_rating:.2f}, Error: {error:.2f}"
    )
