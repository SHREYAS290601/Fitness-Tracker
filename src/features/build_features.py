import pandas as pd
import glob as glob
import sklearn
import matplotlib.pyplot as plt
import pickle as pk
import plotly.express as px
from remove_outliers import (
    remove_outliers_by_IQR,
    plot_binary_outliers,
    mark_outliers_chauvenet,
    local_outlier_analysis,
)
import numpy as np

with open("../../data/final.pickle", "rb") as f:
    final = pk.load(f)

outlier_checks = list(final.columns[:6])

# outlier via normal method
for col in outlier_checks:
    dataset = remove_outliers_by_IQR(final, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# outlier via Chauvents
for col in outlier_checks:
    dataset = mark_outliers_chauvenet(dataset=final, col=col, C=2)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# local outlier analysis
dataset, outliers, X_score = local_outlier_analysis(
    final[outlier_checks], outlier_checks, 30
)
for col in dataset:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


# Specific based checking outlier analysis

# 1 IQR--> Bench
label = "bench"
for col in outlier_checks:
    dataset = remove_outliers_by_IQR(final[final.label == label], col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
# 2 Chauv-->Bench
for col in outlier_checks:
    dataset = mark_outliers_chauvenet(final[final.label == label], col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# Final outlier method use
dataset_final = final.copy()
for col in outlier_checks:
    for label in final.label.unique():
        # per label based understanding outliers
        dataset = mark_outliers_chauvenet(final[final.label == label], col=col)
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # Those who are outliers are then accounted
        dataset_final.loc[(dataset_final.label == label), col] = dataset[col]

        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(
            f"The number of Outliers removed from {col} are {n_outliers} for label {label}"
        )

dataset_final.to_pickle("../../data/interim/final_dataset_outlier_removed.pkl")