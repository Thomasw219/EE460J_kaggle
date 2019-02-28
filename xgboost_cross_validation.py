#!/usr/bin/env python3

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from extract_data import *

np.random.seed(42)

X_train, y_train = get_train_data()
X_test = get_test_data()
depths = [5]
n_estimators = [300, 500, 700]

for depth in depths:
    for n in n_estimators:
        start = time.time()
        print("Model depth " + str(depth) + ", " + str(n) + " estimators")
        grid_params = {
            'max_depth' : [depth],
            'learning_rate' : [0.05, 0.1, 0.5],
            'lambda' : [0.001, 0.005, 0.01],
            'n_jobs' : [7],
            'n_estimators' : [n],
            'eval_metric' : ['auc']
        }
        model = XGBClassifier()
        xgb_grid = GridSearchCV(model, grid_params, cv=3, n_jobs=-1, scoring='roc_auc', return_train_score='warn')
        xgb_grid.fit(X_train, np.ravel(y_train))
        end = time.time()
        print("Time elapsed: " + str(end - start))
        print("Best score: " + str(xgb_grid.best_score_))
        print(xgb_grid.best_params_)
        print("\n\n\n")
        y_test = xgb_grid.predict_proba(X_test)[:, 1:]
        write_predictions(y_test, "xgb_classifier_" + str(depth) + "_" + str(n) + "_fine")
