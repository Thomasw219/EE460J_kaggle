#!/usr/bin/env python3

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
X_train, y_train = outlier_removal(X_train, y_train)
X_test = get_test_data()

name = "xgb_500_5_outlier_removed"
model = XGBClassifier(max_depth=5, n_estimators=500, reg_lambda=5, n_jobs=-1, scale_pos_weight=948/15435, objective='binary:logistic', eval_metric='auc', eta=0.001, subsample=0.5)
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
print("Feature importances: " )
for i, f in enumerate(model.feature_importances_):
    print("f" + str(i + 1) + ": " + str(f))
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, name)
write_model(model, name)
