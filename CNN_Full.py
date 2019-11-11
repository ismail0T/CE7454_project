import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from NN_Utils import *
from my_models import *
from dataloader import SeqDataLoader
from Utils import *
import time, sys
import copy
import netron
data_dir_fpz = "../data_2013/eeg_fpz_cz"
data_dir_eeg = "../data_2013/traindata_eeg"

# data_dir_eog = "../data_2013/EOG_horizontal"
from tsne import bh_sne

# print(data_dir)
classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)

num_epochs = 25
batch_size = 128
learning_rate = 0.001

device = torch.device("cuda:1")


def run_experiment_simple_validation():
    print(' num_epochs: ', num_epochs)
    CNNutils = NN_Utils(learning_rate, batch_size, num_epochs)
    X_train, y_train, X_test, y_test = prep_train_validate_data_CV(data_dir_eeg, 1, batch_size)
    print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
    print('\n')
    char2numY = dict(zip(classes, range(len(classes))))
    for cl in classes:
        print("Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
              len(np.where(y_test == char2numY[cl])[0]))

    # model #
    net = ConvSimpleBest()
    display_num_param(net)
    net = net.to(device)

    train_history, validation_history = CNNutils.train_model_cnn(net, X_train, y_train, X_test, y_test, device)
    print('train_history', train_history)
    print('validation_history', validation_history)

    del net

    plot_one_validation_history(train_history, validation_history)


def run_experiment_cross_validation():
    CNNutils = NN_Utils(learning_rate, batch_size, num_epochs)
    train_history_over_CV = []
    val_history_over_CV = []
    confusion_matrix_train_CV = []
    confusion_matrix_test_CV = []
    num_folds = 20

    print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)


    for fold_id in range(0, num_folds):
        # Loading Data
        # X_train, y_train, X_test, y_test = prep_train_validate_data_no_smote(data_dir_fpz, num_folds, fold_id, batch_size)
        X_train, y_train, X_test, y_test = prep_train_validate_data_CV(data_dir_eeg, fold_id, batch_size)

        if fold_id == 0:
            print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
            # print('Train Data Shape: ', y_train.shape, '  Test Data Shape: ', y_test.shape)

            print('\n')

        # char2numY = dict(zip(classes, range(len(classes))))
        # for cl in classes:
        #     print("__Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
        #           len(np.where(y_test == char2numY[cl])[0]))
        #
        # print("\nFold <" + str(fold_id + 1) + ">")
        # Train Data Shape:  (38912, 1, 3000)   Test Data Shape:  (2048, 1, 3000)
        # Train  W 7305  =>  639
        # Train  N1 2651  =>  74
        # Train  N2 16375  =>  860
        # Train  N3 5270  =>  263
        # Train  REM 7311  =>  212

        # X_2d = bh_sne(X_train)




        sys.exit()
        # model #
        net = ConvSimpleSOTA()
        print("ConvSimpleSOTA..")
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
        print("Acc ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

    plot_CV_history(train_history_over_CV, val_history_over_CV)
    plot_confusion_matrix(confusion_matrix_test_best.data.numpy(), classes)
# plot_confusion_matrix(cm=tt.data.numpy(), target_names=classes)
# from torchsummary import summary
# net = ConvSimpleSOTA().cuda()
# summary(net, (1, 3000))




# val_mlp = np.load("val_MLP_smote.npz")["arr_0"][:, :20]
# val_conv = np.load("val_Conv_smote.npz")["arr_0"][:, :20]
# val_convBatch = np.load("val_ConvBatchNorm_smote.npz")["arr_0"][:, :20]
# val_convLSTM = np.load("val_ConvLSTM_smote.npz")["arr_0"][:, :20]
#
# plot_one_validation_history(val_mlp, val_conv, val_convBatch, val_convLSTM)
#

# val_mlp = np.load("val_ConvLSTM.npz")["arr_0"]
# train_mlp = np.load("train_ConvLSTM.npz")["arr_0"]
#
#
# plot_CV_history(train_mlp, val_mlp)




# print(net)
# tt = torch.from_numpy(np.asarray(
#         [[119,  20,  85,  10,  37],
#         [ 46,  260,  44,   2,  33],
#         [13,  37, 300, 120,  79],
#         [ 21,   1,  88, 183,   1],
#         [18,  58, 134,   8, 235]]))

# plot_confusion_matrix(tt, classes)
# run_experiment_cross_validation()
#
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# print(y.shape)
# X_2d = bh_sne(X)
# scatter(X_2d[:, 0], X_2d[:, 1], c=y)


# get_curr_time()
# ttttt()

# mm = np.asarray([ 35,  100,  63])
# mm.sort()
# print(mm[-2])


# net = ConvLSTM(False)
# torch.save(net, 'model_convLSTM.pth')
# model = torch.load('model0.pth')
# # display_num_param(model)
# netron.start('model0.pth')


# print("Current Time =", current_time)


# net = torch.load('ConvLSTM_best.pt')
# # summary(net.cuda(), (1, 3000))
# top_layer = net.conv1
# filter = top_layer.weight.data.cpu().numpy()
#


# filter = (1/(2 * 3.69201088)) * filter + 0.5
#
#
# # num_cols= choose the grid size you want
# def plot_kernels(tensor, num_cols=8):
#     # if not tensor.ndim == 4:
#     #     raise Exception("assumes a 4D tensor", tensor.ndim)
#     # if not tensor.shape[-1] == 3:
#     #     raise Exception("last dim needs to be 3 to plot")
#     num_kernels = tensor.shape[0]
#     num_rows = 1 + num_kernels // num_cols
#     fig = plt.figure(figsize=(num_cols, num_rows))
#     for i in range(tensor.shape[0]):
#         ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
#         ax1.imshow(tensor[i])
#         ax1.axis('off')
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     plt.show()
#
#
# plot_kernels(filter)


# import visdom
# vis = visdom.Visdom(server='155.69.149.166')
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))

# print(filter)
# plt.imshow(top_layer.weight.data[0][:, :, :, 0].squeeze(), cmap='gray')
# plt.show()




