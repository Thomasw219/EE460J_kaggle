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
X_test = get_test_data()

model = XGBClassifier(n_estimators=200, reg_lambda=1.e-8, n_jobs=-1, booster='gblinear')
results = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, "xgb_500_5")
write_model(model, "xgb_500_5")
