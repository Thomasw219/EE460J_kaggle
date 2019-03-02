#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from joblib import dump

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

def write_predictions(soft_predictions, model_name):
    data = pd.read_csv("data/sample-submission.csv")
    submission = data.drop(columns=['Y'])
    submission.insert(1, 'Y', soft_predictions)
    submission.to_csv(path_or_buf="predictions/" + model_name + ".csv", index=False)

def write_model(model, model_name):
    dump(model, "models/" + model_name + ".joblib")

def scale(X):
    return preprocessing.scale(X)

#get_train_data()
#get_test_data()
#write_predictions([], "test")
