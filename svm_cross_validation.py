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

model = SVC(gamma='scale', probability=True, kernel='poly', degree=2)
results = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
name = "svm_quadratic"
write_predictions(y_test_soft, name)
write_model(model, name)
