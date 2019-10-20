import torch
import torch.nn as nn

from dataloader import SeqDataLoader
import matplotlib.pyplot as plt
import numpy as np


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )


def prep_train_validate_data(data_dir, num_folds, fold_id, batch_size):
    data_loader = SeqDataLoader(data_dir, num_folds, fold_id, classes=5)

    X_train, y_train, X_test, y_test = data_loader.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0], 1)


    total_training_length = (X_train.shape[0] // batch_size) * batch_size
    total_test_length = (X_test.shape[0] // batch_size) * batch_size

    X_train = X_train[:total_training_length]
    y_train = y_train[:total_training_length]
    X_test = X_test[:total_test_length]
    y_test = y_test[:total_test_length]

    return X_train, y_train, X_test, y_test


def plot_CV_history(train_history_over_CV, val_history_over_CV):
        one_fold = train_history_over_CV[0]
        num_epoch = len(one_fold)
        num_fold = len(train_history_over_CV)

        for j in range(0, num_epoch):
            for i in range(1, num_fold):
                train_history_over_CV[0][j] += train_history_over_CV[i][j]

                if j == num_epoch - 1:
                    train_history_over_CV[0][i] /= num_epoch
        train_history_over_CV[0][0] /= num_epoch

        for j in range(0, num_epoch):
            for i in range(1, num_fold):
                val_history_over_CV[0][j] += val_history_over_CV[i][j]

                if j == num_epoch - 1:
                    val_history_over_CV[0][i] /= num_epoch
        val_history_over_CV[0][0] /= num_epoch

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epoch + 1), train_history_over_CV[0])
        plt.plot(range(1, num_epoch + 1), val_history_over_CV[0])
        plt.legend(['Average Train Accuracy', 'Average Test Accuracy'], loc='upper right', prop={'size': 12})
        plt.suptitle('Cross Validation Performance', fontsize=13.0, y=1.08, fontweight='bold')

