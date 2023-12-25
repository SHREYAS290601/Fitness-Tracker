import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Algorithms import ClassificationAlgoriths
import seaborn as sb
import itertools
from sklearn.metrics import max_error, cluster, accuracy_score, confusion_matrix

# Plot settings
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["text.color"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.style.use("fivethirtyeight")

df = pd.read_pickle("../../data/interim/dataset_generated_feature.pkl")

# Splits
df_train = df.drop(["participant", "category", "set"], axis=1)

x = df_train.drop("label", axis=1)
y = df_train.label


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)
x_train.info()

df_train.label.value_counts().plot.bar(label="Total", color="blue")
y_train.value_counts().plot.bar(label="Train", color="dodgerblue")
y_test.value_counts().plot.bar(label="Test", color="royalblue")
plt.legend()
plt.show()


# features used
basic = [
    "acc_x_lowpass",
    "acc_y_lowpass",
    "acc_z_lowpass",
    "gyro_x_lowpass",
    "gyro_y_lowpass",
    "gyro_z_lowpass",
]
square_features = ["acc_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_feature = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse_" in f)]
cluster = [f for f in df_train.columns if "_cluster" in f]


feature_set_1 = list(set(basic))
feature_set_2 = list(set(basic + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_feature))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster))


# Learning
learner = ClassificationAlgoriths()

max_features = 15
selected_features, order_features, order_score = learner.forward_selection(
    max_features, x_train, y_train
)


plt.plot(np.arange(1, max_features + 1, 1), order_score, "-")
plt.xlabel("Features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_set_names = [
    "feature_set_1",
    "feature_set_2",
    "feature_set_3",
    "feature_set_4",
    "selected_features",
]
iterations = 2
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_set_names):
    print("Feature Set: ", i + 1)
    selected_train_x = x_train[possible_feature_sets[i]]
    selected_test_x = x_test[possible_feature_sets[i]]
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\nTrain the Random Forest : ", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_x, y_train, selected_test_x, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)
    performance_test_rf = performance_test_rf / iterations

    print("\n\t Train KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_x, y_train, selected_test_x, gridsearch=True
    )

    performance_test_knn = accuracy_score(y_test, class_test_y)
    print("\n\t Train Decision Tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_x, y_train, selected_test_x, gridsearch=True
    )

    performance_test_decision = accuracy_score(y_test, class_test_y)
    print("\n\t Train Naive Bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_x, y_train, selected_test_x)

    performance_test_naive = accuracy_score(y_test, class_test_y)
    print("\n\t Train Neural Network")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.NeuralNetwork(
        selected_train_x,
        y_train,
        selected_test_x,
        100,
    )
    performance_test_neural = accuracy_score(
        pd.Categorical(y_test).codes, np.argmax(class_test_y, axis=1)
    )

    models = [
        "Random_Forest",
        "Neural_Network_ANN",
        "Decision_Tree",
        "Naive_Bayes",
        "KNN",
    ]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_rf,
                performance_test_neural,
                performance_test_decision,
                performance_test_naive,
                performance_test_knn,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

plt.figure(figsize=(14, 10))
sb.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Models")
plt.ylabel("Accuracy Percentage")
plt.legend()
plt.show()
