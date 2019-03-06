#!/usr/bin/env python3

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X_full, y_full = get_train_data()
X_train, y_train = undersampled_data(2)
X_test = get_test_data()
name = "svm_undersampled_rbf"

model = SVC(C=1, gamma=1.e-10, probability=True)
#model = SVC(kernel='poly', degree=2, probability=True)
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc', n_jobs=-1)
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
model.fit(X_train, y_train)
y_full_proba = model.predict_proba(X_full)[:, 1:]
write_predictions(y_full_proba, name, train=True)
print("full train auc: " + str(roc_auc_score(y_full, y_full_proba)))
y_train_proba = model.predict_proba(X_train)[:, 1:]
print("undersampled train auc: " + str(roc_auc_score(y_train, y_train_proba)))
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, name)
write_model(model, name)
