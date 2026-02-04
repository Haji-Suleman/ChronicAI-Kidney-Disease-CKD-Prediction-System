# %%
# Date: 2026-02-04
# Author: Haji Suleman
# Project: 🩺 Chronic Kidney Disease (CKD) Prediction System


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./kidney_disease.csv")

df.head()

# %%
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# %%
df["classification"] = df["classification"].str.strip()
df["classification"].value_counts()
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

# %%
df.isnull().sum()

# %%
df.shape

# %%
df = df.dropna()
df.shape

# %%
df.head()

# %%
# Check 'rbc' column
df["rbc"].value_counts()

# %%
df["rbc"] = df["rbc"].map({"normal": 1, "abnormal": 0})
df["rbc"].value_counts()

# %%
# Check 'pc' column
df["pc"].value_counts()

# %%
df["pc"] = df["pc"].map({"normal": 1, "abnormal": 0})
df["pc"].value_counts()

# %%
# Check 'pcc' column
df["pcc"].value_counts()

# %%
df["pcc"] = df["pcc"].map({"present": 1, "notpresent": 0})
df["pcc"].value_counts()

# %%
# Check 'ba' column
df["ba"].value_counts()

# %%
df["ba"] = df["ba"].map({"present": 1, "notpresent": 0})
df["ba"].value_counts()

# %%
# Check 'htn' column
df["htn"].value_counts()

# %%
df["htn"] = df["htn"].map({"yes": 1, "no": 0})
df["htn"].value_counts()

# %%
# Check 'dm' column
df["dm"].value_counts()

# %%
df["dm"] = df["dm"].map({"yes": 1, "no": 0})
df["dm"].value_counts()

# %%
# Check 'cad' column
df["cad"].value_counts()

# %%
df["cad"] = df["cad"].map({"yes": 1, "no": 0})
df["cad"].value_counts()

# %%
# Check 'appet' column
df["appet"].value_counts()

# %%
df["appet"] = df["appet"].map({"good": 1, "poor": 0})
df["appet"].value_counts()

# %%
# Check 'pe' column
df["pe"].value_counts()

# %%
df["pe"] = df["pe"].map({"yes": 1, "no": 0})
df["pe"].value_counts()

# %%
# Check 'ane' column
df["ane"].value_counts()

# %%
df["ane"] = df["ane"].map({"yes": 1, "no": 0})
df["ane"].value_counts()

print(df[:5])

# %%
