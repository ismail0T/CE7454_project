import os

import torch
import torch.nn as nn

from dataloader import SeqDataLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )


def prep_train_validate_data(data_2013_folder, batch_size):
    max_time_step = 10

    x = np.load(data_2013_folder + "/trainData__SMOTE_all_10s_f11.npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_2013_folder + "/trainData__SMOTE_all_10s_f11_TEST.npz")
    X_test = x2["x"]
    y_test = x2["y"]

    # print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
    # print('Train y_train Shape: ', y_train.shape, '  Test y_test Shape: ', y_test.shape)

    X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]
    # print('\nTrain Data Shape: ', X_train.shape)

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]

    X_train = np.reshape(X_train, [-1, 1, 3000])
    y_train = np.reshape(y_train, [-1, 1])
    X_test = np.reshape(X_test, [-1, 1, 3000])
    y_test = np.reshape(y_test, [-1, 1])

    X_train = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    y_train = y_train[:(X_train.shape[0] // batch_size) * batch_size]
    X_test = X_test[:(X_test.shape[0] // batch_size) * batch_size, :]
    y_test = y_test[:(y_test.shape[0] // batch_size) * batch_size]

    return X_train, y_train, X_test, y_test


def prep_train_validate_data_CV(data_dir, fold_id, batch_size):
    max_time_step = 10

    x = np.load(data_dir + "/trainData__SMOTE_all_10s_f"+ str(fold_id) +".npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_dir + "/trainData__SMOTE_all_10s_f"+ str(fold_id) +"_TEST.npz")
    X_test = x2["x"]
    y_test = x2["y"]

    # print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
    # print('Train y_train Shape: ', y_train.shape, '  Test y_test Shape: ', y_test.shape)

    X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]
    # print('\nTrain Data Shape: ', X_train.shape)

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]

    X_train = np.reshape(X_train, [-1, 1, 3000])
    y_train = np.reshape(y_train, [-1, 1])
    X_test = np.reshape(X_test, [-1, 1, 3000])
    y_test = np.reshape(y_test, [-1, 1])

    X_train = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    y_train = y_train[:(X_train.shape[0] // batch_size) * batch_size]
    X_test = X_test[:(X_test.shape[0] // batch_size) * batch_size, :]
    y_test = y_test[:(y_test.shape[0] // batch_size) * batch_size]

    return X_train, y_train, X_test, y_test


def plot_CV_history(train_history_over_CV, val_history_over_CV):
    one_fold = np.copy(train_history_over_CV[0])
    num_epoch = len(one_fold)

    train = np.asarray(np.matrix(train_history_over_CV).mean(0)).reshape(-1)
    test = np.asarray(np.matrix(val_history_over_CV).mean(0)).reshape(-1)

    max_val = max(test)
    print("Epoch:", np.argmax(test), " max_acc=", max_val)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train)
    plt.plot(range(1, num_epoch + 1), test)
    plt.legend(['Average Train Accuracy', 'Average Test Accuracy'], loc='upper right', prop={'size': 12})
    plt.suptitle('Cross Validation Performance', fontsize=15.0, y=1.08, fontweight='bold')
    plt.title('Best accuracy: ' + '{0:.2f}'.format(max_val) + ' % (epoch: ' + str(np.argmax(test)) + ')', fontsize=13.0,
              y=1.08, fontweight='bold')

    plt.show()


def plot_one_validation_history(train_history, val_history):
    num_epoch = len(train_history)


    max_val = max(val_history)

    # print(num_epoch)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train_history)
    plt.plot(range(1, num_epoch + 1), val_history)
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right', prop={'size': 12})
    plt.suptitle('Global Performance', fontsize=15.0, y=1.08, fontweight='bold')
    plt.title('Best accuracy: ' + '{0:.2f}'.format(max_val) + ' % (epoch: ' + str(np.argmax(val_history)) + ')', fontsize=13.0,
              y=1.08, fontweight='bold')

    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    print(cm)
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





