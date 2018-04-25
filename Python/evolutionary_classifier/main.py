from evo_classifier.LinearOEC import LinearOEC
from sklearn.svm import LinearSVC
import time

import numpy as np
import pandas as pd


# Breast Cancer dataset: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# data = pd.read_csv('../../Data/BC.csv')


# HillVally: http://archive.ics.uci.edu/ml/datasets/Hill-Valley
data = pd.read_csv('../../Data/HV.csv')

# ATTENTION: In both datasets, the null space has been removed and the data has been normalized to 0 mean and 1 variance.
# This makes the HV problem harder to solve!

data_array = data.as_matrix()
classes = data_array[:,data.shape[1] - 1]
data_array = np.delete(data_array, -1, axis=1)

# SVM for comparison
start_time = time.time()
model = LinearSVC()
model.fit(data_array, classes)
SVMTime = time.time() - start_time

preds = model.predict(data_array)
print("SVM: total loss was ", sum(abs(preds-classes)), " over ", np.shape(data_array)[0], " instances. Time: ", SVMTime*1000.0, " (ms)")

# Build the OEC model and fit it
start_time = time.time()
# classifier = LinearOEC(np.shape(data_array)[1], optimizer='cmaes',regularization=0.0)
classifier = LinearOEC(np.shape(data_array)[1], optimizer='cmaes',regularization=0.0,
                       initialization=model.coef_[0], iterations=10)
classifier.fit(data_array, classes)
OECTime = time.time() - start_time

# Predict classes
classes_test = classifier.predict(data_array)

print("OEC: total loss was ", sum(abs(classes_test-classes)), " over ", np.shape(data_array)[0], " instances. Time: ", OECTime*1000.0, " (ms)")

