#!/usr/bin/env python3

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

from extract_data import *

def stack_soft(X, models):
    stack_columns = np.empty((len(X), 0))
    for model_name in models:
        model = load("models/" + model_name + ".joblib")
        y_hat = model.predict_proba(X)[:, 1:]
        stack_columns = np.append(stack_columns, y_hat, axis=1)
    return np.append(X, stack_columns, axis=1)

def stack_hard(X, models):
    stack_columns = np.empty((len(X), 0))
    for model_name in models:
        model = load("models/" + model_name + ".joblib")
        y_hat = model.predict(X)
        stack_columns = np.append(stack_columns, y_hat, axis=1)
    return np.append(X, stack_columns, axis=1)

def interaction_term(X, input_columns, funcs):
    for k, func in enumerate(funcs):
        new_col = np.empty((0,1))
        for i in range(X.shape[0]):
            args = []
            for j in input_columns[k]:
                args.append(X[i][j])
            args_tup = tuple(args)
            new_col = np.append(new_col, np.array([func(*args_tup)]).reshape(1,1), axis=0)
        X = np.append(X, new_col, axis=1)
    return X

def stack_from_files(X, filenames, train=False):
    for filename in filenames:
        path = "predictions/" + filename + ".csv"
        if train:
            path = "predictions/" + filename + "_train.csv"
        data = pd.read_csv(path).values
        predictions = data[:, 1:]
        X = np.append(X, predictions, axis=1)
    return X

def show_correlations():
    data = pd.read_csv("data/train_final.csv")
    zeros = data.loc[data[data.Y == 0].index]
    ones = data.loc[data[data.Y == 1].index]
    corr = zeros.corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.show()
    corr = ones.corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.show()
