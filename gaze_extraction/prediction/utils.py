import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from tqdm import tqdm

import os
import matplotlib.pyplot as plt


def k_fold_parameter_search(
    X_train,
    y_train,
    model,
    parameters_to_tune,
    num_splits=10,
    use_oversampling=False,  # whether to do oversampling before training
    scaling=None,  # whether to do scaling before training
    random_seed=42,
    score_fn=f1_score,
    show_result=False,
    return_dict=False,
):
    """
    Perform k-fold cross validation to find the best parameters for the model.
    For each parameter combination, we do k-fold cross validation and return the average score (default f1).
    """

    def show_k_fold_result(val_results):
        for key, item in val_results.items():
            print("Parameters Set: ", key)
            print(f"Validation F1-Score:{item:.4f}")
            print("----------------------------")

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    # kf = StratifiedKFold(
    #     n_splits=num_splits, shuffle=True, random_state=random_seed
    # )  # use stratified k-fold to ensure the same distribution of labels in each fold
    if use_oversampling:
        # First we find the indexes the categorical features
        cate_features = find_categorical_features(X_train)
        # use SMOTENC for oversampling
        # sm = SMOTENC(categorical_features=cate_features, random_state=random_seed)
        sm = SMOTE(random_state=random_seed)
    val_results = dict()
    # holding out different validation sets:
    for train_index, val_index in kf.split(X_train, y_train):
        # split the dataframe
        train_x, val_x = X_train.iloc[train_index], X_train.iloc[val_index]
        train_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]
        if use_oversampling:
            train_x, train_y = sm.fit_resample(train_x, train_y)
        if scaling:
            train_x = scaling.fit_transform(train_x)
            val_x = scaling.transform(val_x)
        for parameters in tqdm(parameters_to_tune, desc="Tuning parameters"):
            # print(parameters)
            # training
            clf = model(**parameters)
            clf.fit(train_x, train_y)

            # validation
            val_predictions = clf.predict(val_x)
            val_score = score_fn(val_y, val_predictions)
            # update the score dict
            val_results[tuple(parameters.values())] = (
                val_results.get(tuple(parameters.values()), 0) + val_score
            )
    for key in val_results.keys():
        val_results[key] /= num_splits
    if show_result:
        best_parameters = max(val_results, key=val_results.get)
        show_k_fold_result(val_results)
        print("Best parameters: ", best_parameters)
        print("Best score: ", val_results[best_parameters] / num_splits)
    if return_dict:
        return val_results
    return max(val_results, key=val_results.get)


def find_categorical_features(X, threshold=6):
    """
    Locate the categorical features in the data. We consider a feature to be categorical if it has less than threshold unique values.
    """
    categorical_features = []
    for i, feature in enumerate(X.T):
        if len(np.unique(feature)) <= threshold:
            categorical_features.append(i)
    return categorical_features


def plot_validation_result(
    val_results, x_var_index, common_var_name, x_var_name, model_name
):
    """
    Plot the validation result for different Sets of parameters.
    val_results: the validation results from k_fold_parameter_search
    x_var_index: the index of the variable that we want to plot on the x-axis
    common_var_name: the name of the variable to be shown on the legend
    x_var_name: the name of the variable to be shown on the x-axis
    model_name: the name of the model
    """
    # assume the variable other than x should be the one on the legend (only 2 in all)
    common_var_index = 1 - x_var_index
    line_dct = dict()
    x_labels = []
    # enumerate all the parameters, sorted by x axis variable
    for key, item in sorted(val_results.items(), key=lambda x: x[0][x_var_index]):
        # if the x axis variable is not in the x_labels, add it
        if key[x_var_index] not in x_labels:
            x_labels.append(key[x_var_index])
        # Deal with special case for AdaBoost, where the parameter stored would be a base estimator (model)
        if hasattr(key[common_var_index], "max_depth"):
            common_key = key[common_var_index].max_depth
        elif hasattr(key[common_var_index], "C"):
            common_key = key[common_var_index].C
        else:
            common_key = key[common_var_index]
        line_dct[common_key] = line_dct.get(common_key, []) + [item]

    # plot the result
    for common_var, scores in line_dct.items():
        plt.plot(x_labels, scores, label=f"{common_var_name}={common_var}", marker="o")
    plt.legend()
    plt.xlabel(x_var_name)
    plt.xticks(x_labels)
    plt.ylabel("F1-Score")
    plt.title("Validation Performance for Different Hyperparameters for " + model_name)
    plt.show()
