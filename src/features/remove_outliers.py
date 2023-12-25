import pandas as pd
import typing
from typing import Tuple, Tuple, Optional, Any
import math
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

plt.style.use(["seaborn-v0_8-bright"])
plt.rcParams["figure.figsize"] = [12, 8]


def remove_outliers_by_IQR(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.

Args:
    data (pd.DataFrame): The DataFrame containing the data.
    col (str): The name of the column to remove outliers from.

Returns:
    pd.DataFrame: A new DataFrame with the outliers removed.

Examples:
    >>> data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> remove_outliers_by_IQR(data, 'col1')
       col1  col1_outlier
    0     1         False
    1     2         False
    2     3         False
    3     4         False
    4     5         False
    5     6         False
    6     7         False
    7     8         False
    8     9         False
    9    10         False
"""
    dataset = data.copy()

    quartile1 = dataset[col].quantile(0.25)
    quartile3 = dataset[col].quantile(0.75)

    IQR = quartile3 - quartile1

    lower_bound = quartile1 - (1.5 * IQR)
    upper_bound = quartile3 + (1.5 * IQR)

    dataset[col + "_outlier"] = (dataset[col] > upper_bound) | (
        dataset[col] < lower_bound
    )
    return dataset


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()
    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


clf = LocalOutlierFactor(n_neighbors=30, contamination="auto")


def local_outlier_analysis(
    dataset: pd.DataFrame, columns: str, n: int
) -> Tuple[pd.DataFrame, np.array, np.array]:
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores
