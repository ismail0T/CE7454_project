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


def prep_train_validate_data(data_2013_folder, batch_size):
    max_time_step = 10

    x = np.load(data_2013_folder + "/trainData__SMOTE_all_10s_f0.npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_2013_folder + "/trainData__SMOTE_all_10s_f0_TEST.npz")
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


def prep_train_validate_data_CV(data_dir, num_folds, fold_id, batch_size):
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

        # print(train_history_over_CV[0])
        # print(val_history_over_CV[0])

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epoch + 1), train_history_over_CV[0])
        plt.plot(range(1, num_epoch + 1), val_history_over_CV[0])
        plt.legend(['Average Train Accuracy', 'Average Test Accuracy'], loc='upper right', prop={'size': 12})
        plt.suptitle('Cross Validation Performance', fontsize=13.0, y=1.08, fontweight='bold')
        plt.show()


def plot_one_validation_history(train_history, val_history):
    num_epoch = len(train_history)

    # print(num_epoch)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train_history)
    plt.plot(range(1, num_epoch + 1), val_history)
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right', prop={'size': 12})
    plt.suptitle('Global Performance', fontsize=13.0, y=1.08, fontweight='bold')
    plt.show()



