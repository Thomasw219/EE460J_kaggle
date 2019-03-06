#!/usr/bin/env python3

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from extract_data import *

np.random.seed(42)

X_train, y_train = undersampled_data(3)
X_test = get_test_data()

model = SVC()
grid_params = {
    'C' : [0.1, 0.5, 1, 5, 10],
    'probability' : [True],
    'gamma' : [1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4],
}
xgb_grid = GridSearchCV(model, grid_params, cv=5, n_jobs=-1, scoring='roc_auc', return_train_score='warn')
xgb_grid.fit(X_train, np.ravel(y_train))
print("Best score: " + str(xgb_grid.best_score_))
print(xgb_grid.best_params_)
name = "svm_rbf_gamma_" + str(xgb_grid.best_params_['gamma'])

y_test = xgb_grid.predict_proba(X_test)[:, 1:]
write_predictions(y_test, name)
write_model(model, name)
