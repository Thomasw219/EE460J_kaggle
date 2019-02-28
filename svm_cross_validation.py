#!/usr/bin/env python3

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

X_train, y_train = get_train_data()
X_test = get_test_data()

model = SVC()
grid_params = {
    'C' : [0.25, 0.5, 1, 2]
}
xgb_grid = GridSearchCV(model, grid_params, cv=3, n_jobs=-1, scoring='roc_aoc', return_train_score='warn')
xgb_grip.fit(X_train, np.ravel(y_train))

y_test = xgb_grid.predict_proba(X_test)[:, 1:]
write_predictions(y_test, "svm_rbf")
