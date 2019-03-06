#!/usr/bin/env python3

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
stack = ["xgb_500_5_undersampled", "scaled_knn"]
X_train = stack_soft(X_train, stack)
X_test = stack_soft(get_test_data(), stack)

name = "lr_stack"
for n in stack:
    name += "_" + n
model = LogisticRegression(solver='lbfgs')
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
end = time.time()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_train)[:, 1:]
write_predictions(y_proba, name, train=True)
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
print("Train auc: " + str(roc_auc_score(y_train, y_proba)))
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, name)
write_model(model, name)
