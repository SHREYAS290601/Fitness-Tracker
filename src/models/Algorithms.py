from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from typing import Tuple
from tensorflow.keras.layers import Dropout, Softmax, Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import copy


class ClassificationAlgoriths:
    def forward_selection(
        self, max_features: int, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[list, list, list]:
        """
        Perform forward feature selection to choose the right features by combining them one after another.

        Args:
            max_features: The maximum number of features to select.
            x_train: The input training data as a pandas DataFrame.
            y_train: The target training data as a pandas Series.

        Returns:
            A tuple containing three lists:
            - selected_features: The selected features.
            - order_features: The order in which the features were selected.
            - order_score: The accuracy score corresponding to each selected feature.

        Raises:
            None.
        """
        order_features = []
        order_score = []
        selected_features = []
        ca = ClassificationAlgoriths()
        prev_best_pref = 0

        # this is a feature selection method of choosing the right features by combining them one after other
        for i in range(0, max_features):
            print(i)
            features_left = list(set(x_train.columns) - set(selected_features))
            best_pref = 0
            best_attributes = ""

            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

            # now we get the accuracy and see if the feature is better or the combination it is a bit of exhaustive process
            (
                pred_y_train,
                pred_y_test,
                prob_training_y,
                prob_test_y,
            ) = ca.decision_tree(
                x_train[temp_selected_features],
                y_train,
                x_train[temp_selected_features],  # this acts as the base for x_test
            )
            pref = accuracy_score(y_train, pred_y_train)

            # now we compare
            if pref > best_pref:
                best_pref = pref
                best_feature = f

            selected_features.append(f)
            prev_best_pref = best_feature
            order_features.append(best_feature)
            order_score.append(best_pref)
        return selected_features, order_features, order_score

    def NeuralNetwork(self,x_train, y_train, x_test, epochs=100):
        """
        Train a neural network model using the given training data and predict the labels for the test data.

        Args:
            x_train: The input training data.
            y_train: The target training data.
            x_test: The input test data.
            epochs: The number of epochs to train the model.

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        classes = y_train.nunique()
        model = Sequential()
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(classes, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(
            x_train,
            pd.Categorical(y_train).codes,
            epochs=epochs,
            batch_size=(len(x_train) // epochs),
            verbose=0,
        )

        pred_prob_train_y = model.predict_proba(x_train)
        pred_prob_test_y = model.predict_proba(x_test)
        pred_train_y = model.predict(x_train)
        pred_test_y = model.predict(x_test)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_train_y, columns=y_train.unique()
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=y_train.unique())
        return pred_train_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    def support_vector_machine_with_kernel(
        self,
        train_X,
        train_y,
        test_X,
        kernel="rbf",
        C=1,
        gamma=1e-3,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Train a support vector machine model with a specified kernel on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.
            kernel: The kernel function to be used (default: "rbf").
            C: The regularization parameter (default: 1).
            gamma: The kernel coefficient (default: 1e-3).
            gridsearch: Whether to perform grid search for hyperparameter tuning (default: True).
            print_model_details: Whether to print the best parameters if grid search is performed (default: False).

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None."""
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"kernel": ["rbf", "poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(
                SVC(probability=True), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            svm = SVC(
                C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000
            )

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def support_vector_machine_without_kernel(
        self,
        train_X,
        train_y,
        test_X,
        C=1,
        tol=1e-3,
        max_iter=1000,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Train a support vector machine model without a kernel on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.
            C: The regularization parameter (default: 1).
            tol: The tolerance for stopping criteria (default: 1e-3).
            max_iter: The maximum number of iterations (default: 1000).
            gridsearch: Whether to perform grid search for hyperparameter tuning (default: True).
            print_model_details: Whether to print the best parameters if grid search is performed (default: False).

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"max_iter": [1000, 2000], "tol": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring="accuracy")
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        # this technique is a replication found on stackoverflow
        distance_training_platt = 1 / (1 + np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = (
            distance_training_platt / distance_training_platt.sum(axis=1)[:, None]
        )
        distance_test_platt = 1 / (1 + np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = (
            distance_test_platt / distance_test_platt.sum(axis=1)[:, None]
        )
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    def k_nearest_neighbor(
        self,
        train_X,
        train_y,
        test_X,
        n_neighbors=5,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Train a k-nearest neighbors model on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.
            n_neighbors: The number of neighbors to consider (default: 5).
            gridsearch: Whether to perform grid search for hyperparameter tuning (default: True).
            print_model_details: Whether to print the best parameters if grid search is performed (default: False).

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [{"n_neighbors": [1, 2, 5, 10]}]
            knn = GridSearchCV(
                KNeighborsClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def decision_tree(
        self,
        train_X,
        train_y,
        test_X,
        min_samples_leaf=50,
        criterion="gini",
        print_model_details=False,
        export_tree_path="../visualization/trees",
        export_tree_name="tree.dot",
        gridsearch=True,
    ):
        """
        Train a decision tree model on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.
            min_samples_leaf: The minimum number of samples required to be at a leaf node (default: 50).
            criterion: The function to measure the quality of a split (default: "gini").
            print_model_details: Whether to print the best parameters and feature importances if grid search is performed (default: False).
            export_tree_path: The path to export the decision tree visualization (default: "Example_graphs/Chapter7/").
            export_tree_name: The name of the exported decision tree visualization file (default: "tree.dot").
            gridsearch: Whether to perform grid search for hyperparameter tuning (default: True).

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "criterion": ["gini", "entropy"],
                }
            ]
            dtree = GridSearchCV(
                DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, criterion=criterion
            )

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=dtree.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        # code is from another author
        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(dtree.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            print("Feature importance decision tree:")
            for i in range(0, len(dtree.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(dtree.feature_importances_[ordered_indices[i]])
            tree.export_graphviz(
                dtree,
                out_file=export_tree_path + export_tree_name,
                feature_names=train_X.columns,
                class_names=dtree.classes_,
            )

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X):
        """
        Train a naive Bayes model on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        # Create the model
        nb = GaussianNB()

        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def random_forest(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=10,
        min_samples_leaf=5,
        criterion="gini",
        print_model_details=False,
        gridsearch=True,
    ):
        """
        Train a random forest model on the given training data and predict the labels for the test data.

        Args:
            train_X: The input training data.
            train_y: The target training data.
            test_X: The input test data.
            n_estimators: The number of trees in the forest (default: 10).
            min_samples_leaf: The minimum number of samples required to be at a leaf node (default: 5).
            criterion: The function to measure the quality of a split (default: "gini").
            print_model_details: Whether to print the best parameters and feature importances if grid search is performed (default: False).
            gridsearch: Whether to perform grid search for hyperparameter tuning (default: True).

        Returns:
            A tuple containing the following:
            - pred_training_y: The predicted labels for the training data.
            - pred_test_y: The predicted labels for the test data.
            - frame_prob_training_y: The predicted probabilities for each class for the training data.
            - frame_prob_test_y: The predicted probabilities for each class for the test data.

        Raises:
            None.
        """
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                }
            ]
            rf = GridSearchCV(
                RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True
                )
            ]
            print("Feature importance random forest:")
            for i in range(0, len(rf.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(rf.feature_importances_[ordered_indices[i]])

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )
