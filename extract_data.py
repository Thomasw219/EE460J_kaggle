#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

def get_train_data():
    data = pd.read_csv("data/train_final.csv");
    mat = data.values
    X = mat[:, 2:]
    y = mat[:, 1:2]
    return (X,y)

def get_test_data():
    data = pd.read_csv("data/test_final.csv")
    print(data.head())
    mat = data.values
    X = mat[:, 1:]
    return X

def write_predictions(soft_predictions, model_name):
    data = pd.read_csv("data/sample-submission.csv")
    print(data)


#get_train_data()
#get_test_data()
