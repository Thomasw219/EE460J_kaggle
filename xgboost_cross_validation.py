#!/usr/bin/env python3

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from extract_data import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
X_test = get_test_data()

model = XGBClassifier(max_depth=5, n_estimators=500, reg_lambda=0.000001, n_jobs=-1)
results = cross_val_score(model, X_train, y_train, cv=8, scoring='roc_auc')
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, "xgb_cv_soft")
y_test_hard = model.predict(X_test)
write_predictions(y_test_hard, "xgb_cv_hard")
write_model(model, "xgb_500_5")
