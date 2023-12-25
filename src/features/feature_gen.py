import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from Helper import data_transform, temporal_abstraction
from Helper import fourier_transform
import seaborn as sb
from sklearn.cluster import KMeans

df = pd.read_pickle("../../data/interim/final_dataset_outlier_removed.pkl")

predictor_columns = list(df.columns[:6])

# * Plotting *#
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["text.color"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.style.use("fivethirtyeight")

# Now we have outliers being removed, so to fill the gaps we use interpolate
df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# Examine the medium, heavy and standing or sitting ones with time plots
df[df.set == 25].acc_x.plot()  # heavy 5
df[df.set == 50].acc_x.plot()  # medium 10

duration = df[df.set == 1].index[-1] - df[df.set == 1].index[0]
print(f"Duration of workout is {duration.seconds}s for {df[df.set==1].category[0]}")

for single in df.set.unique():
    start = df[df.set == single].index[0]
    stop = df[df.set == single].index[-1]

    duration = stop - start

    df.loc[df.set == single, "duration"] = duration.seconds

df.groupby("category").duration.mean().plot.line()

# * Apply low pass *#
df_low = df.copy()
lowpass = data_transform.LowPass()
sf = 1000 / 200
cutoff = 1.25
for col in predictor_columns:
    df_low[col] = lowpass.low_pass_filter(
        df_low, col=col, sampling_frequency=sf, cutoff_frequency=cutoff
    )[col]

for cols in df_low.columns[:6]:
    df_low[df_low.set == 45][cols].plot.line(subplots=True, title=str.upper(cols))
    plt.show()
    df_low[df_low.set == 45][cols + "_lowpass"].plot.line(
        subplots=True, title=str.upper(cols + "_Lowpass")
    )
    plt.show()

# * PCA *#
PCA = data_transform.PrincipalComponentAnalysis()
df_pca = df_low.copy()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.plot(pc_values)

predictor_columns = [col + "_lowpass" for col in predictor_columns]
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
df_pca

# PCA VISUALIZATION
visual = df_pca[df_pca.set == 45]
visual[["pca_1", "pca_2", "pca_3"]].plot()


# Sum of square
""" Basically we need r which is the scalar magnitude of the x,y,z components of the acc or gyro
 So to do this we use simply the formula r=(x^2+y^2+z^2)^0.5
"""
df_squared = df_pca.copy()

acc_r = (
    df_squared.acc_x_lowpass**2
    + df_squared.acc_y_lowpass**2
    + df_squared.acc_z_lowpass**2
)
gyro_r = (
    df_squared.gyro_x_lowpass**2
    + df_squared.gyro_y_lowpass**2
    + df_squared.gyro_z_lowpass**2
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

df_squared[df_squared.set == 45][["acc_r", "gyro_r"]].plot(subplots=True)

# Temporal Abstraction #

df_temporal = df_squared.copy().iloc[:, 9:]
df_temporal = df_temporal.merge(
    df[["participant", "label", "category"]],
    how="inner",
    left_index=True,
    right_index=True,
)
Nbs = temporal_abstraction.NumericalAbstraction()

window_size = ws = 1000 // 200

predictor_columns += ["acc_r", "gyro_r"]
predictor_columns
#! Why use the unique? Basically in out df there are 4 previous values of squat say and next 4 are of benchpress, so the earlier values are used by window to calc the mean/std whatever.So we have to avoid this !#
df_temporal_subset_combo = []
for s in df_temporal.set.unique():
    subset = df_temporal[df_temporal.set == s].copy()
    for col in predictor_columns:
        subset = Nbs.abstract_numerical(subset, [col], ws, "mean")
        subset = Nbs.abstract_numerical(subset, [col], ws, "std")
        subset = Nbs.abstract_numerical(subset, [col], ws, "median")
    df_temporal_subset_combo.append(subset)
df_temporal_copy = df_temporal.copy()
df_temporal_copy = pd.concat(df_temporal_subset_combo)

#! plot !#
df_temporal_copy[df_temporal_copy.set == 71][
    [
        "gyro_z_lowpass",
        "gyro_z_lowpass_temp_mean_ws_5",
        "gyro_z_lowpass_temp_median_ws_5",
        "gyro_z_lowpass_temp_std_ws_5",
    ]
].plot()
df_temporal_copy.info()

# Frequency abstraction using FFT
df_freq = df_temporal_copy.copy()
df_freq = df_freq.reset_index()
FreqAbs = fourier_transform.FourierTransformation()
fs = 1000 // 200
ws = 2800 // 200
df_freq_try = FreqAbs.abstract_frequency(df_freq, ["acc_y_lowpass"], ws, fs)
df_freq_try[df_freq_try.set == 15].acc_y_lowpass.plot()
df_freq_try[df_freq_try.set == 15][
    [
        "acc_y_lowpass_max_freq",
        "acc_y_lowpass_freq_weighted",
        "acc_y_lowpass_pse",
        "acc_y_lowpass_freq_1.429_Hz_ws_14",
        "acc_y_lowpass_freq_1.786_Hz_ws_14",
        "acc_y_lowpass_freq_2.5_Hz_ws_14",
    ]
].plot()
df_freq
df_freq_list = []
for s in df_freq.set.unique():
    print(f"The current set is {s}")
    subset = df_freq[df_freq.set == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
df_freq_final = pd.concat(df_freq_list)
df_freq_final = df_freq_final.set_index("epoch (ms)", drop=True)

df_freq_final


# Deaaling with overlap
df_freq_final_dropped = df_freq_final.dropna()
df_cluster = df_freq_final_dropped.iloc[::2]

# we will use the cluster number as a feature as well

k_values = range(2, 10)
acc_intertia = []

for k in k_values:
    subset = df_cluster[["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"]]
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20)
    cluster_labels_acc = kmeans.fit_predict(subset)
    acc_intertia.append(kmeans.inertia_)

pd.Series(acc_intertia).plot()
pd.Series(cluster_labels_acc).value_counts().plot.bar()


gyro_intertia = []

for k in k_values:
    subset = df_cluster[["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"]]
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20)
    cluster_labels_gyro = kmeans.fit_predict(subset)
    gyro_intertia.append(kmeans.inertia_)

pd.Series(gyro_intertia).plot()
pd.Series(cluster_labels_gyro).value_counts().plot.bar()

# taking k as 5
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=20)

subset = df_cluster[["acc_x_lowpass", "acc_y_lowpass", "acc_z_lowpass"]]
df_cluster["acc_cluster"] = kmeans.fit_predict(subset)
df_cluster.acc_cluster.value_counts().plot.bar()

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20)
subset = df_cluster[["gyro_x_lowpass", "gyro_y_lowpass", "gyro_z_lowpass"]]
df_cluster["gyro_cluster"] = kmeans.fit_predict(subset)
df_cluster.gyro_cluster.value_counts().plot.bar()

df_cluster
#3d rep
fig=plt.figure(figsize=(20,20))
ax=fig.add_subplot(projection="3d")
for cl in df_cluster.acc_cluster.unique():
    subset=df_cluster[df_cluster.acc_cluster==cl]
    ax.scatter(subset["acc_x_lowpass"],subset["acc_y_lowpass"],subset["acc_z_lowpass"],label=cl)
ax.set_xlabel("X")
ax.set_xlabel("Y")
ax.set_xlabel("Z")
plt.legend()
plt.show()
fig=plt.figure(figsize=(20,20))
ax=fig.add_subplot(projection="3d")
for cl in df_cluster.gyro_cluster.unique():
    subset=df_cluster[df_cluster.gyro_cluster==cl]
    ax.scatter(subset["acc_x_lowpass"],subset["acc_y_lowpass"],subset["acc_z_lowpass"],label=cl)
ax.set_xlabel("X")
ax.set_xlabel("Y")
ax.set_xlabel("Z")
plt.legend()
plt.show()

df_cluster.to_pickle("../../data/interim/dataset_generated_feature.pkl")