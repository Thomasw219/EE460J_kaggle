#!/usr/bin/env python3

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
X_test = get_test_data()

model = SVC(gamma='scale', probability=True)
results = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, "svm_stacked")
write_model(model, "stacked_svm")
