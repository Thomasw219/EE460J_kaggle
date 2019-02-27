#!/usr/bin/env python3

import numpy as np
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV

X_train, y_train = get_train_data()
grid_params = {
    'max_depth' : [2, 4, 6, 8, 10]
    'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    'lambda' : [0.1, 0.5, 1, 5, 10]
    'n_jobs' : [4]
    'n_estimators' : [50, 100, 200, 300, 400, 500]
    'eval_metric' : ['auc']
}

model = XGBClassifier()
xgb_grid = GridSearchCV(model, grid_params, cv=5, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print(xgb_grid.best_params_)
