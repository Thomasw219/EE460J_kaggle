#!/usr/bin/env python3

import numpy as np
from joblib import load

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
