#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from joblib import dump
from imblearn.over_sampling import ADASYN, SMOTE


def get_train_data():
    data = pd.read_csv("data/train_final.csv");
    mat = data.values
    X = mat[:, 2:]
    y = mat[:, 1:2]
    return (X,np.ravel(y))

def get_test_data():
    data = pd.read_csv("data/test_final.csv")
    mat = data.values
    X = mat[:, 1:]
    return X

def write_predictions(soft_predictions, model_name, train=False):
    data = pd.read_csv("data/sample-submission.csv")
    if not train:
        submission = data.drop(columns=['Y'])
        submission.insert(1, 'Y', soft_predictions)
        submission.to_csv(path_or_buf="predictions/" + model_name + ".csv", index=False)
    else:
        df = pd.DataFrame(data={'Y' : np.ravel(soft_predictions)}, dtype='float64')
        df.to_csv(path_or_buf="predictions/" + model_name + "_train.csv", index=True)


def write_model(model, model_name):
    dump(model, "models/" + model_name + ".joblib")

def scale(X):
    return preprocessing.scale(X)

def outlier_removal(X, y):
    X_aug = np.append(X, y.reshape(y.shape[0], 1), axis=1)
    model = IsolationForest(behaviour='new', contamination="auto")
    ones = np.empty((0, X.shape[1] + 1))
    zeros = np.empty((0, X.shape[1] + 1))
    for i in range(X.shape[0]):
        x_i = np.append(X[i], np.array([y[i]]), axis=0)
        if y[i] == 1:
            ones = np.append(ones, np.array([x_i]), axis=0)
        else:
            zeros = np.append(zeros, np.array([x_i]), axis=0)
    outlier_ones = model.fit_predict(ones)
    outlier_zeros = model.fit_predict(zeros)
    removed_ones = np.array([ones[i] for i in range(outlier_ones.shape[0]) if outlier_ones[i] == 1 ])
    removed_zeros = np.array([zeros[i] for i in range(outlier_zeros.shape[0]) if outlier_zeros[i] == 1])
    X_aug_removed = np.append(removed_ones, removed_zeros, axis=0)
    print(X_aug_removed.shape)
    X_new = X_aug_removed[:, :X_aug_removed.shape[1] - 1]
    y_new = np.ravel(X_aug_removed[:, X_aug_removed.shape[1] - 1:])
    print(X_new)
    print(y_new)
    return X_new, y_new

def undersampled_data(proportion):
    data = pd.read_csv("data/train_final.csv")
    zeros = data[data.Y == 0].index
    ones = data[data.Y == 1].index
    print(len(zeros))
    print(len(ones))
    print(len(zeros) + len(ones))
    random_ones = np.random.choice(ones, round(proportion * len(zeros)), replace=False)
    mat = np.append(data.loc[zeros], data.loc[random_ones], axis=0)
    X = mat[:, 2:]
    y = mat[:, 1:2]
    return (X,np.ravel(y))

def oversampling_SMOTE():
    resampler = SMOTE(random_state=42, n_neighbors=10)
    X, y = get_train_data()
    return resampler.fit_resample(X, y)

def oversampling_ADASYN():
    resampler = ADASYN(n_neighbors=10)
    X, y = get_train_data()
    return resampler.fit_resample(X, y)

def one_hot_encoding(features):
    data = pd.read_csv("data/train_final.csv")
    one_hot_data = pd.get_dummies(data, columns=features)
    mat = one_hot_data.values
    X = mat[:, 2:]
    y = mat[:, 1:2]
    return (X,np.ravel(y))
