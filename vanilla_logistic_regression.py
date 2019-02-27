#!/usr/bin/env python3

import extract_data
import numpy as np
from sklearn import linear_model.LogisticRegression

X, y = get_train_data()
model = LogisticRegression()
model.fit(X, y)

