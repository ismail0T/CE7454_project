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

num_epochs = 10
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
    num_folds = 20

    print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)


    for fold_id in range(0, num_folds):
        # Loading Data
        X_train, y_train, X_test, y_test = prep_train_validate_data_CV(data_dir, fold_id, batch_size)

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
        net = ConvSimpleBest()
        if fold_id == 0:
            display_num_param(net)
        net = net.to(device)

        train_history, validation_history = CNNutils.train_model_cnn(net, X_train, y_train, X_test, y_test, device)
        # print('train_history', train_history)
        # accumulate history for each CV loop, then take average
        train_history_over_CV.append(train_history)
        val_history_over_CV.append(validation_history)

        del net

    print(train_history_over_CV)
    print(val_history_over_CV)

    plot_CV_history(train_history_over_CV, val_history_over_CV)


# train_history =[53.07816247002398, 74.08011091127098, 79.10858812949641, 83.64808153477217, 86.69439448441247, 89.55523081534771, 91.04841127098321, 93.36967925659472, 94.03851918465229, 95.79398980815348]
# validation_history = [60.50646551724138, 61.341594827586206, 59.348060344827594, 58.16271551724138, 63.03879310344828, 61.449353448275865, 62.365301724137936, 65.32866379310344, 67.88793103448276, 65.40948275862068]

#
# plot_one_validation_history(train_history, validation_history)

# run_experiment_cross_validation()


# train_y_batch = np.random.uniform(0, 5, 10000)
# predicted = np.random.uniform(0, 5, 10000)
#
# confusion_matrix = torch.zeros(5, 5, dtype=torch.int32)
#
# stacked = torch.stack(
#                     (
#                         torch.from_numpy(train_y_batch).int()
#                         , torch.from_numpy(predicted).int()
#                     )
#                     , dim=1
#                 )
#
# for p in stacked:
#     tl, pl = p.tolist()
#     confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1
#
# print(confusion_matrix.long().diag())
# print(confusion_matrix.long().sum(1))
# confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy())*100
#
# print(confusion_matrix_accuracy)
#
# for i, cl in enumerate(classes):
#     print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')


# plot_confusion_matrix(confusion_matrix, classes)

# run_experiment_simple_validation()
# SeqDataLoader.save_to_npz_file(None, train_history, validation_history, 1, "file_name")


run_experiment_cross_validation()
# print('ffff')
#
#
# # train = [[58.94725678733032, 74.81617647058823, 78.82494343891403], [54.90056818181818, 73.4268465909091, 77.17329545454545], [57.798165137614674, 73.49125573394495, 77.72720756880734]]
# # test  = [[62.50723379629629, 74.10300925925925, 71.94733796296296], [68.48480504587155, 70.7497133027523, 76.21129587155964], [70.41713169642857, 77.53208705357143, 77.21121651785714]]
# # np.savez('train.npz', train)
# # np.savez('test.npz', test)
#
# #
# #
#
# with np.load('train.npz') as f:
#
#     train_history_over_CV = f
# #
#     print(len(train_history_over_CV[0]))
# # print(val_history_over_CV.shape)
#
# # plot_CV_history(train, test)
