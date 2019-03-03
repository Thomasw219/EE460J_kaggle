#!/usr/bin/env python3

import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score

from extract_data import *

np.random.seed(42)

start = time.time()
X_train, y_train = get_train_data()
X_train = scale(X_train)
X_test = scale(get_test_data())

model = LDA()
results = cross_val_score(model, X_train, y_train, cv=8, scoring='roc_auc')
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, "lda_cv")
write_model(model, "vanilla_lda")
