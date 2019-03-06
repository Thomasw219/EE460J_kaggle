#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from extract_data import *
from feature_engineering import *

X, y_train = get_train_data()
mult = lambda x, y : x * y
diff = lambda x, y : x - y
ratio = lambda x, y : x / y
square = lambda x : x * x
#cols = [[0,13],[0,14],[13,14]]
cols = [[i] for i in range(24)]
#funcs = [ratio, ratio, ratio]
funcs = [square for i in range(24)]
X_train = interaction_term(scale(X), cols, funcs)
#X_train = scale(X)
name = "lr_squared_features"

model = LogisticRegression(solver='lbfgs', max_iter=1000)
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_train)[:, 1:]
write_predictions(y_proba, name, train=True)

print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
print("Train auc: " + str(roc_auc_score(y_train, y_proba)))
print("Feature importances: " )
for i, f in enumerate(model.coef_[0]):
    print("f" + str(i + 1) + ": " + str(f))

X_test = interaction_term(scale(get_test_data()), cols, funcs)
y_test = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test, name)
write_model(model, name)
