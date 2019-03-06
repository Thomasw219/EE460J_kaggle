#!/usr/bin/env python3

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from extract_data import *
from feature_engineering import *

X_train, y_train = get_train_data()
name = "isolation_forest"

model = IsolationForest(behaviour='new', contamination=0.06)
inlier = model.fit_predict(X_train)
write_predictions(inlier, name, train=True)

X_test = get_test_data()
inlier_test = model.predict(X_test)
write_predictions(inlier_test, name)
write_model(model, name)
