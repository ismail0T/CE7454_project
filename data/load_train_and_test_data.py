"""
After loading the data,
X_train shape should be (5343, 10, 3000)
"""
import numpy as np


max_time_step=10

x = np.load("E:/data_2013_npz/traindata3/trainData__SMOTE_all_10s_f0.npz")
X_train = x["x"]
y_train = x["y"]

x2 = np.load("E:/data_2013_npz/traindata3/trainData__SMOTE_all_10s_f0_TEST.npz")
X_test = x2["x"]
y_test = x2["y"]


X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]

X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
y_train = np.reshape(y_train,[-1,y_test.shape[1],])

# shuffle training data_2013
permute = np.random.permutation(len(y_train))
X_train = np.asarray(X_train)
X_train = X_train[permute]
y_train = y_train[permute]
