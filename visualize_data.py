#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from extract_data import *

X, y = get_train_data()
X = scale(X)

for i in range(X.shape[1]):
    x_i = X[:, i:i+1]
    plt.figure("f" + str(i + 1))
    plt.hist(x_i)
    plt.show()
    plt.figure("logf" + str(i + 1))
    plt.hist(scale(np.log1p(np.log1p(np.sum([np.ones((len(x_i), 1)), x_i], axis=0)))))
    plt.show()
