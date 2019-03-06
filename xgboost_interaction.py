#!/usr/bin/env python3

import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X, y_train = get_train_data()
X_test = get_test_data()
mult = lambda x, y : x * y
diff = lambda x, y : x - y
ratio = lambda x, y : x / y
cols = [[0,13],[0,14],[13,14]]
funcs = [ratio, ratio, ratio]
X_train = interaction_term(scale(X), cols, funcs)

model = XGBClassifier(max_depth=5, n_estimators=500, reg_lambda=0.01, n_jobs=-1)
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')

end = time.time()
model.fit(X_train, y_train)
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
print("Feature importances: " )
for i, f in enumerate(model.feature_importances_):
    print("f" + str(i + 1) + ": " + str(f))
#y_test_soft = model.predict_proba(X_test)[:, 1:]
#write_predictions(y_test_soft, "xgb_500_5")
#write_model(model, "xgb_500_5")
