#!/usr/bin/env python3

import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from extract_data import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
X_train = scale(X_train)
X_test = scale(get_test_data())
NN = KNC(n_neighbors=150)

model = Pipeline([("Scaler", StandardScaler()), ("KNearestNeighbors", NN)])
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
end = time.time()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_train)[:, 1:]
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
print("Train auc: " + str(roc_auc_score(y_train, y_proba)))
y_test_soft = model.predict_proba(X_test)[:, 1:]
name = "scaled_knn"
write_predictions(y_proba, name, train=True)
write_predictions(y_test_soft, name)
write_model(model, name)
