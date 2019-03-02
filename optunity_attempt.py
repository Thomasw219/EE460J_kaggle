#!/usr/bin/env python3

import numpy as np
import optunity
import optunity.metrics
import sklearn.svm

from extract_data import *

X_train, y_train = get_train_data()
X_train = scale(X_train)
X_test = scale(get_test_data())

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=X_train, y=y_train, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-7, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X_train, y_train)
y_test = optimal_model.predict_proba(X_test)[:, 1:]
write_predictions(y_test, "optunity_svm")
write_model(optimal_model, "optunity_svm")
