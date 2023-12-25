import pandas as pd
import glob as glob


# * DF creation for acc and gyro *#

files = glob.glob("../../data/raw/MetaMotion/MetaMotion/*.csv")


def create_dataset(files):
    """
    Create a dataset by combining accelerometer and gyroscope data from multiple files.

    Args:
        files: A list of file paths to read the data from.

    Returns:
        Two DataFrames: `acc_final` containing the accelerometer data and `gyro_final` containing the gyroscope data.

    Raises:
        None."""

    data_path = "../../data/raw/MetaMotion/MetaMotion"
    acc = pd.DataFrame()
    gyro = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")[-1]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if bool(f.count("Acc")):
            df["set"] = acc_set
            acc_set += 1
            acc = pd.concat([acc, df])
        else:
            df["set"] = gyro_set
            gyro_set += 1
            gyro = pd.concat([gyro, df])

    acc.index = pd.to_datetime(acc["epoch (ms)"], unit="ms")
    gyro.index = pd.to_datetime(gyro["epoch (ms)"], unit="ms")

    acc_final = acc.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)
    gyro_final = gyro.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)
    return acc_final, gyro_final


acc, gyro = create_dataset(files)

data_merge = pd.concat([acc.iloc[:, :3], gyro], axis=1)

data_merge.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set",
]


#! Frequency Conversion !#

conversion = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

##This step is a bit confusing.Basically if we do resampling we get the values based on the number of days that we are looking for
# while this is good it can potentially blow up our dataset to something like nonsense
# so , to solve it we divide the day based on days , curretnly a week, and resample plus recombine

days = [g for n, g in data_merge.groupby(pd.Grouper(freq="D"))]

# resample + rejoin
final = pd.concat([df.resample(rule="200ms").apply(conversion).dropna() for df in days])

final.set = final.set.astype("int")


import pickle as pk

pk.dump(final, open("final.pickle", "wb"))
