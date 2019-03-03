#!/usr/bin/env python3

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
stack = ["vanilla_lr", "scaled_knn"]
X_train = stack_soft(X_train, stack)
X_test = stack_soft(get_test_data(), stack)

model = XGBClassifier(max_depth=5, n_estimators=500, reg_lambda=0.01, n_jobs=-1)
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
name = "xgb_stack"
for n in stack:
    name += "_" + n
write_predictions(y_test_soft, name)
write_model(model, name)
