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

data_dir = "../data_2013_npz"

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

    # model #
    net = ConvSimple()
    display_num_param(net)
    net = net.to(device)

    train_history, validation_history = CNNutils.train_model(net, X_train, y_train, X_test, y_test, device)
    print('train_history', train_history)
    print('validation_history', validation_history)

    del net

    plot_one_validation_history(train_history, validation_history)





def run_experiment_cross_validation():
    CNNutils = CNN_Utils(learning_rate, batch_size, num_epochs)
    train_history_over_CV = []
    val_history_over_CV = []
    num_folds = 10

    print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)

    for fold_id in range(0, num_folds):
        # Loading Data
        X_train, y_train, X_test, y_test = prep_train_validate_data(data_dir, num_folds, fold_id, batch_size)

        if fold_id == 0:
            print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
            print('\n')
        print("\nFold <" + str(fold_id+1) + ">")

        # model #
        net = ConvSimple()
        if fold_id == 0:
            display_num_param(net)
        net = net.to(device)

        train_history, validation_history = CNNutils.train_model(net, X_train, y_train, X_test, y_test, device)
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
#
# plot_one_validation_history(train_history, validation_history)

run_experiment_simple_validation()
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
