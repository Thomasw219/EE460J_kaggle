#!/usr/bin/env python3

import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score

from extract_data import *
from feature_engineering import *

np.random.seed(42)

start = time.time()
X, y_train = get_train_data()
log = lambda x : np.log1p(x)
cols = [[0]]
X_train = interaction_term(X, cols, [log])
X_test = interaction_term(get_test_data(), cols, [log])

model = LDA()
results = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
end = time.time()
print("Time elapsed: " + str(end - start))
print("Scores: " + str(results))
print("Mean: " + str(np.mean(results)))
print("Std: " + str(np.std(results)))
model.fit(X_train, y_train)
y_test_soft = model.predict_proba(X_test)[:, 1:]
write_predictions(y_test_soft, "lda_cv")
write_model(model, "vanilla_lda")
