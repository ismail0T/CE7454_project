import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, repeat
import itertools

from dataloader import SeqDataLoader


def print_confusion_matrix_accuracy(confusion_matrix_test_best, classes):
    confusion_matrix_accuracy = (confusion_matrix_test_best.diag().numpy() / confusion_matrix_test_best.sum(
        1).numpy()) * 100
    for i, cl in enumerate(classes):
        print("Acc ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )

def prep_train_validate_data_no_smote_double_channel(data_dir1, data_dir2, num_folds, fold_id, batch_size):
    X_train1, y_train1, X_test1, y_test1 = prep_train_validate_data_no_smote(data_dir1, num_folds, fold_id, batch_size)
    X_train2, y_train2, X_test2, y_test2 = prep_train_validate_data_no_smote(data_dir2, num_folds, fold_id, batch_size)

    min_length = X_train1.shape[0] if X_train1.shape[0] < X_train2.shape[0] else X_train2.shape[0]
    # min_length = 1000
    X_train1 = X_train1[:min_length, :, :]
    X_train2 = X_train2[:min_length, :, :]

    y_train1 = y_train1[:min_length, :]
    y_train2 = y_train2[:min_length, :]

    X_train = np.concatenate((X_train1, X_train2), axis=1)
    y_train = np.concatenate((y_train1, y_train2), axis=1)

    min_length_test = X_test1.shape[0] if X_test1.shape[0] < X_test2.shape[0] else X_test2.shape[0]
    # min_length_test = 1000
    X_test1 = X_test1[:min_length_test, :, :]
    X_test2 = X_test2[:min_length_test, :, :]

    y_test1 = y_test1[:min_length_test, :]
    y_test2 = y_test2[:min_length_test, :]

    X_test = np.concatenate((X_test1, X_test2), axis=1)
    y_test = np.concatenate((y_test1, y_test2), axis=1)

    return X_train, y_train, X_test, y_test


def prep_train_validate_data_no_smote(data_dir, num_folds, fold_id, batch_size):
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


def prep_train_validate_data_CV(data_dir, fold_id, batch_size, seq_len=1):
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

    # X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    # y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]
    # y_train = y_train[]

    # X_train = np.array([X_train[i:i + seq_len] for i in range(0, len(X_train), seq_len)])
    y_train = np.array(list(chain.from_iterable(zip(*repeat(y_train, seq_len)))))
    y_test = np.array(list(chain.from_iterable(zip(*repeat(y_test, seq_len)))))

    X_train = np.reshape(X_train, [-1, 1, 3000//seq_len])
    y_train = np.reshape(y_train, [-1, 1])
    X_test = np.reshape(X_test, [-1, 1, 3000//seq_len])
    y_test = np.reshape(y_test, [-1, 1])

    # print('3_Train Data Shape: ', X_train.shape, '  y_train Data Shape: ', y_train.shape)
    # print('4_Test Data Shape: ', X_test.shape, '  y_test Data Shape: ', y_test.shape)

    # print(X_train[1][0], '\n')

    X_train = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    y_train = y_train[:(X_train.shape[0] // batch_size) * batch_size]
    X_test = X_test[:(X_test.shape[0] // batch_size) * batch_size, :]
    y_test = y_test[:(y_test.shape[0] // batch_size) * batch_size]


    return X_train, y_train, X_test, y_test

def prep_train_validate_data_CV_double_channel(data_dir1, data_dir2, fold_id, batch_size, seq_len):
    X_train1, y_train1, X_test1, y_test1 = prep_train_validate_data_CV(data_dir1, fold_id, batch_size, seq_len=1)
    X_train2, y_train2, X_test2, y_test2 = prep_train_validate_data_CV(data_dir2, fold_id, batch_size, seq_len=1)

    min_length = X_train1.shape[0] if X_train1.shape[0] < X_train2.shape[0] else X_train2.shape[0]
    min_length = 1000
    X_train1 = X_train1[:min_length, :, :]
    X_train2 = X_train2[:min_length, :, :]

    y_train1 = y_train1[:min_length, :]
    y_train2 = y_train2[:min_length, :]

    X_train = np.concatenate((X_train1,X_train2), axis=1)
    y_train = np.concatenate((y_train1, y_train2), axis=1)

    min_length_test = X_test1.shape[0] if X_test1.shape[0] < X_test2.shape[0] else X_test2.shape[0]
    min_length_test = 1000
    X_test1 = X_test1[:min_length_test, :, :]
    X_test2 = X_test2[:min_length_test, :, :]

    y_test1 = y_test1[:min_length_test, :]
    y_test2 = y_test2[:min_length_test, :]

    X_test = np.concatenate((X_test1, X_test2), axis=1)
    y_test = np.concatenate((y_test1, y_test2), axis=1)

    return X_train, y_train, X_test, y_test

def prep_train_validate_data_CV_seq2seq(data_dir, fold_id, batch_size, seq_len, classes):
    char2numY = dict(zip(classes, range(len(classes))))
    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
    char2numY['<SOD>'] = len(char2numY)
    char2numY['<EOD>'] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))

    max_time_step = 10

    x = np.load(data_dir + "/trainData__SMOTE_all_10s_f"+ str(fold_id) +".npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_dir + "/trainData__SMOTE_all_10s_f"+ str(fold_id) +"_TEST.npz")
    X_test = x2["x"]
    y_test = x2["y"]

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    X_test = [X_test[i:i + seq_len] for i in range(0, len(X_test), seq_len)]
    y_test = [y_test[i:i + seq_len] for i in range(0, len(y_test), seq_len)]

    if X_test[-1].shape[0] != seq_len:
        X_test.pop()
        y_test.pop()

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)



    X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]

    # print('X_train Data Shape: ', X_train.shape, '  y_train Data Shape: ', y_train.shape)
    # print('X_test Data Shape:  ', X_test.shape, '   y_test Data Shape:  ', y_test.shape)
    #
    # print('\n')

    # add '<SOD>' to the beginning of each label sequence, and '<EOD>' to the end of each label sequence (both for training and test sets)
    y_train = [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_train]
    y_train = np.array(y_train)

    y_test = [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_test]
    y_test = np.array(y_test)






    # print('x Data Shape: ', X_train.shape, '  x2 Data Shape: ', y_train.shape)
    # print('Train y_train Shape: ', y_train.shape, '  Test y_test Shape: ', y_test.shape)

    # X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    # y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]
    # print('\nTrain Data Shape: ', X_train.shape)

    # X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    # y_train = np.reshape(y_train, [-1, y_test.shape[1], ])

    # shuffle training data_2013
    # permute = np.random.permutation(len(y_train))
    # X_train = np.asarray(X_train)
    # X_train = X_train[permute]
    # y_train = y_train[permute]
    # y_train = y_train[]

    # X_train = np.array([X_train[i:i + seq_len] for i in range(0, len(X_train), seq_len)])
    # y_train = np.array(list(chain.from_iterable(zip(*repeat(y_train, seq_len)))))
    # y_test = np.array(list(chain.from_iterable(zip(*repeat(y_test, seq_len)))))

    # X_train = np.reshape(X_train, [-1, 1, 3000])
    # y_train = np.reshape(y_train, [-1, 1])
    # X_test = np.reshape(X_test, [-1, 1, 3000])
    # y_test = np.reshape(y_test, [-1, 1])

    # print('3_Train Data Shape: ', X_train.shape, '  y_train Data Shape: ', y_train.shape)
    # print('4_Test Data Shape: ', X_test.shape, '  y_test Data Shape: ', y_test.shape)

    # print(X_train[1][0], '\n')

    # X_train = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    # y_train = y_train[:(X_train.shape[0] // batch_size) * batch_size]
    # X_test = X_test[:(X_test.shape[0] // batch_size) * batch_size, :]
    # y_test = y_test[:(y_test.shape[0] // batch_size) * batch_size]


    return X_train, y_train, X_test, y_test


def plot_CV_history(train_history_over_CV, val_history_over_CV):
    one_fold = np.copy(train_history_over_CV[0])
    num_epoch = len(one_fold)

    train = np.asarray(np.matrix(train_history_over_CV).mean(0)).reshape(-1)
    test = np.asarray(np.matrix(val_history_over_CV).mean(0)).reshape(-1)

    max_val = max(test)
    print("Epoch:", np.argmax(test)+1, " max_acc=", max_val)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train)
    plt.plot(range(1, num_epoch + 1), test)
    plt.legend(['Average Train Accuracy', 'Average Test Accuracy'], loc='upper right', prop={'size': 12})
    plt.suptitle('Cross Validation Performance', fontsize=15.0, y=1.08, fontweight='bold')
    plt.title('Best accuracy: ' + '{0:.2f}'.format(max_val) + ' % (epoch: ' + str(np.argmax(test)+1) + ')', fontsize=13.0,
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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        my_marks = [-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        classes = ['', target_names[0], '', target_names[1], '', target_names[2], '', target_names[3], '', target_names[4]]
        plt.yticks(my_marks, classes)

        plt.xticks(tick_marks, target_names, rotation=45)

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f} %".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
