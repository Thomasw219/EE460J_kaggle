#!/usr/bin/env python3

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from extract_data import *

np.random.seed(42)

X_train, y_train = get_train_data()
X_test = get_test_data()

model = SVC()
grid_params = {
    'C' : [5, 10, 20],
    'probability' : [True],
    'gamma' : ['scale', 'auto'],
    'decision_function_shape' : ['ovo']
}
xgb_grid = GridSearchCV(model, grid_params, cv=3, n_jobs=-1, scoring='roc_auc', return_train_score='warn')
xgb_grid.fit(X_train, np.ravel(y_train))
print("Best score: " + str(xgb_grid.best_score_))
print(xgb_grid.best_params_)

y_test = xgb_grid.predict_proba(X_test)[:, 1:]
write_predictions(y_test, "svm_rbf_c" + str(xgb_grid.best_params_['C']))
