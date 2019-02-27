#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import LogisticRegression

from extract_data import *

X_train, y_train = get_train_data()
model = LogisticRegression()
model.fit(X_train, np.ravel(y_train))
X_test = get_test_data()
y_test = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test, "vanilla_logistic_regression")
