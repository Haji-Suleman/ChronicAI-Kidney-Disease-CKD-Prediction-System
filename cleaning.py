# cleaning.py
# Date: 2026-02-04
# Author: Haji Suleman
# Project: 🩺 Chronic Kidney Disease (CKD) Prediction System

import pandas as pd
import numpy as np


def clean_ckd_dataset(file_path="./kidney_disease.csv") -> pd.DataFrame:
    # Load dataset
    df = pd.read_csv(file_path)

    # Strip extra spaces
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Clean target column
    df["classification"] = df["classification"].str.strip()
    df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

    # Drop rows with missing values
    df = df.dropna()

    # Map all categorical columns to numeric
    df["rbc"] = df["rbc"].map({"normal": 1, "abnormal": 0})
    df["pc"] = df["pc"].map({"normal": 1, "abnormal": 0})
    df["pcc"] = df["pcc"].map({"present": 1, "notpresent": 0})
    df["ba"] = df["ba"].map({"present": 1, "notpresent": 0})
    df["htn"] = df["htn"].map({"yes": 1, "no": 0})
    df["dm"] = df["dm"].map({"yes": 1, "no": 0})
    df["cad"] = df["cad"].map({"yes": 1, "no": 0})
    df["appet"] = df["appet"].map({"good": 1, "poor": 0})
    df["pe"] = df["pe"].map({"yes": 1, "no": 0})
    df["ane"] = df["ane"].map({"yes": 1, "no": 0})

    # Return cleaned dataframe
    return df


# Make the cleaned df available directly if the file is imported
cleaned_df = clean_ckd_dataset()
