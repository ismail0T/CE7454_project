import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from CNN_Utils import *
from my_models import *
from dataloader import SeqDataLoader
from Utils import *
import time, sys
import copy

data_dir = "../data_2013/new"

classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)

num_epochs = 15
batch_size = 128
learning_rate = 0.001

device = torch.device("cuda:1")


def run_experiment_simple_validation():
    print(' num_epochs: ', num_epochs)
    CNNutils = CNN_Utils(learning_rate, batch_size, num_epochs)
    X_train, y_train, X_test, y_test = prep_train_validate_data(data_dir, batch_size)
    print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
    print('\n')
    char2numY = dict(zip(classes, range(len(classes))))
    for cl in classes:
        print("Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
              len(np.where(y_test == char2numY[cl])[0]))

    # model #
    net = ConvSimple()
    display_num_param(net)
    net = net.to(device)

    train_history, validation_history = CNNutils.train_model_cnn(net, X_train, y_train, X_test, y_test, device)
    print('train_history', train_history)
    print('validation_history', validation_history)

    del net

    plot_one_validation_history(train_history, validation_history)


def run_experiment_cross_validation():
    CNNutils = CNN_Utils(learning_rate, batch_size, num_epochs)
    train_history_over_CV = []
    val_history_over_CV = []
    confusion_matrix_train_CV = []
    confusion_matrix_test_CV = []
    num_folds = 3

    print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)


    for fold_id in range(0, num_folds):
        # Loading Data
        X_train, y_train, X_test, y_test = prep_train_validate_data_CV(data_dir, fold_id, batch_size, seq_len=1)
        if fold_id == 0:
            print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
            print('\n')

        char2numY = dict(zip(classes, range(len(classes))))
        for cl in classes:
            print("__Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
                  len(np.where(y_test == char2numY[cl])[0]))

        print("\nFold <" + str(fold_id + 1) + ">")
        # Train Data Shape:  (38912, 1, 3000)   Test Data Shape:  (2048, 1, 3000)
        # Train  W 7305  =>  639
        # Train  N1 2651  =>  74
        # Train  N2 16375  =>  860
        # Train  N3 5270  =>  263
        # Train  REM 7311  =>  212

        # sys.exit()
        # model #
        net = MLP()
        print("MLP..")
        if fold_id == 0:
            display_num_param(net)
        net = net.to(device)

        train_history, validation_history, confusion_matrix_train_list, confusion_matrix_test_list = CNNutils.train_model_cnn(net, X_train, y_train, X_test, y_test, device)
        # print('train_history', train_history)
        # accumulate history for each CV loop, then take average
        train_history_over_CV.append(train_history)
        val_history_over_CV.append(validation_history)
        confusion_matrix_train_CV.append(confusion_matrix_train_list)
        confusion_matrix_test_CV.append(confusion_matrix_test_list)
        # print(confusion_matrix_test)

        del net

    print(train_history_over_CV)
    print(val_history_over_CV)

    confusion_matrix_test_CV_final = confusion_matrix_test_CV[0]
    for i in range(1, num_folds):
        confusion_matrix_test_CV_final += confusion_matrix_test_CV[i]

    best_epoch_id = np.argmax(np.asarray(np.matrix(val_history_over_CV).mean(0)).reshape(-1))

    confusion_matrix_test_best = confusion_matrix_test_CV_final[best_epoch_id]

    confusion_matrix_accuracy = (confusion_matrix_test_best.diag().numpy() / confusion_matrix_test_best.sum(1).numpy()) * 100
    for i, cl in enumerate(classes):
        print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

    plot_CV_history(train_history_over_CV, val_history_over_CV)
    plot_confusion_matrix(confusion_matrix_test_best.data.numpy(), classes)
# plot_confusion_matrix(cm=tt.data.numpy(), target_names=classes)


# tt = torch.from_numpy(np.asarray(
#         [[119,  20,  85,  10,  37],
#         [ 46,  260,  44,   2,  33],
#         [13,  37, 300, 120,  79],
#         [ 21,   1,  88, 183,   1],
#         [18,  58, 134,   8, 235]]))

# plot_confusion_matrix(tt, classes)
run_experiment_cross_validation()
# ttttt()