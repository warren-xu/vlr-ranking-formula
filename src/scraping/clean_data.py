# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: clean_data.py - used to clean scraped data from vlr.
# Warren Xu

import os
import pandas as pd


def clean_percentage(column):
    """Remove percentage signs and convert to float."""
    return column.str.rstrip("%").astype(float)


csv_files = [file for file in os.listdir("data/raw") if file.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join("data/raw", file)

    df = pd.read_csv(file_path)

    # Clean cols with percentages
    percentage_columns = ["KAST", "HS%", "CL%", "FKPR", "FDPR"]
    for col in percentage_columns:
        if col in df.columns:
            df[col] = clean_percentage(df[col])

    # KAST, FKPR and FDPR are used as per-round rates, not percentages
    for col in ["KAST", "FKPR", "FDPR"]:
        df[col] = df[col] / 100

    # Calculates DPR
    if "D" in df.columns and "Rnd" in df.columns:
        df["DPR"] = round(df["D"] / df["Rnd"], 3)

    # Calculate ADRa
    if "ADR" in df.columns and "Rnd" in df.columns and "K" in df.columns:
        totalDamage = df["ADR"] * df["Rnd"]
        damagePerKill = 140 * df["K"]
        ADRa = (totalDamage - damagePerKill) / df["Rnd"]
        df["ADRa"] = round(ADRa, 3)

    # Calculate survival rating
    if "Rnd" in df.columns and "D" in df.columns:
        survivalRating = (df["Rnd"] - df["D"]) / df["Rnd"]
        df["SR"] = round(survivalRating, 3)

    # Remove rows where data is missing
    required_columns = [
        "Player",
        "R2.0",
        "ACS",
        "K:D",
        "KAST",
        "ADR",
        "KPR",
        "APR",
        "FKPR",
        "FDPR",
        "HS%",
        "CL%",
        "K",
        "D",
        "A",
        "FK",
        "FD",
    ]
    df.dropna(subset=required_columns, inplace=True)

    # Some matches (mostly 2023 China league) lack full round data, so VLR's first-kill
    # count disagrees with its own FK% and the rating doesn't follow the usual formula.
    # Drop players whose FK count and FK% differ by more than 2 percentage points.
    inconsistent = (df["FK"] / df["Rnd"] - df["FKPR"]).abs() > 0.02
    df = df[~inconsistent]
    print(f"{file}: dropped {inconsistent.sum()} players with incomplete match data")

    # Remove unnecessary cols
    df.drop(columns=["Rnd", "CL", "KMax", "Team"], inplace=True)

    df.to_csv(f"data/clean/{file}", index=False)

    print(f"Cleaned data saved to data/clean/{file}")

print("\nData cleaning process for all files completed.")
