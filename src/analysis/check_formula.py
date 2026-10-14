# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: check_formula.py - tests formula on csv file to validate results.
# Warren Xu
#
# Usage: python src/analysis/check_formula.py [csv] [player ...]
#   defaults to data/clean/champions2023.csv and every player in it.

import json
import sys

import pandas as pd

# Written by analyze_data.py, so this always uses the latest fit.
with open("data/formula.json") as f:
    formula = json.load(f)


def calculate_vlr_rating(stats):
    rating = sum(coef * stats[key] for key, coef in formula["coefficients"].items())
    return rating + formula["intercept"]


csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/clean/champions2023.csv"
players = {name.lower() for name in sys.argv[2:]}

df = pd.read_csv(csv_file)
if players:
    df = df[df["Player"].str.lower().isin(players)]

player_name_width = df["Player"].str.len().max() + 2

errors = []
for _, row in df.iterrows():
    actual_rating = row["R2.0"]
    predicted_rating = calculate_vlr_rating(row)
    error = actual_rating - predicted_rating
    errors.append(abs(error))

    player_name = row["Player"].ljust(player_name_width)
    print(
        f"{player_name} | Actual: {actual_rating:.2f}, Predicted: {predicted_rating:.2f}, Error: {error:+.2f}"
    )

print(f"\n{len(errors)} players | mean absolute error: {sum(errors) / len(errors):.3f}")
