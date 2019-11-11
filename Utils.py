import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, repeat
import itertools

import torch
import os
from NN_Utils import NN_Utils
from dataloader_v2 import SeqDataLoader
import copy
import re, datetime, operator, logging, sys
import numpy as np
from collections import namedtuple


classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)
path_to_saved_models = "saved_models"


def run_experiment_cross_validation(model, data_dir, num_folds, num_epochs, learning_rate, batch_size, device, do_balance=False, sampling=0, do_print=True):
    MyNNutils = NN_Utils(learning_rate, batch_size, num_epochs)
    train_history_over_CV = []
    val_history_over_CV = []
    confusion_matrix_train_CV = []
    confusion_matrix_test_CV = []

    print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)
    init_state = copy.deepcopy(model.state_dict())
    best_accuracy = 0

    for fold_id in range(0, num_folds):
        model.load_state_dict(init_state)

        if sampling == 0:  # no modification
            X_train, y_train, X_test, y_test = prep_train_validate_data_no_smote(data_dir, num_folds, fold_id, batch_size)
        elif sampling == 1:  # over-sampling
            X_train, y_train, X_test, y_test = prep_train_validate_data_CV(data_dir, fold_id, batch_size)
        else:  # under-sampling
            X_train, y_train, X_test, y_test = prep_train_validate_data_CV_RUS(data_dir, fold_id, batch_size)

        if do_print:
            if fold_id == 0:
                print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
                print('\n')

            char2numY = dict(zip(classes, range(len(classes))))
            for cl in classes:
                print("__Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
                      len(np.where(y_test == char2numY[cl])[0]))

        print("\nFold <" + str(fold_id + 1) + ">", get_curr_time())

        # model #
        net = model
        if fold_id == 0:
            display_num_param(net)
        net = net.to(device)

        # train_history = validation_history = [[], []]
        # confusion_matrix_train_list = confusion_matrix_test_list = [[], [], [], []]
        if model.__class__.__name__ == "MLP" or model.__class__.__name__ == "ConvSimple" or model.__class__.__name__ == 'ConvBatchNorm':
            train_history, validation_history, confusion_matrix_train_list, confusion_matrix_test_list = MyNNutils.train_model_cnn(
                net, X_train, y_train, X_test, y_test, device, do_balance=do_balance, do_print=do_print)
        else:
            train_history, validation_history, confusion_matrix_train_list, confusion_matrix_test_list = MyNNutils.train_model_conv_lstm(
                net, X_train, y_train, X_test, y_test, False, device, do_balance=do_balance, do_print=do_print)

        # accumulate history for each CV loop, then take average
        train_history_over_CV.append(train_history)
        val_history_over_CV.append(validation_history)
        confusion_matrix_train_CV.append(confusion_matrix_train_list)
        confusion_matrix_test_CV.append(confusion_matrix_test_list)

        curr_acc = max(validation_history)
        if curr_acc > best_accuracy:
            best_accuracy = best_accuracy
            torch.save(net, os.path.join(path_to_saved_models,model.__class__.__name__ + '_best.pt'))

        del net

    if do_print:
        print(train_history_over_CV)
        print(val_history_over_CV)
        print('\n')

    # print(confusion_matrix_test_CV.shape)
    best_epoch_id = np.argmax(np.asarray(np.matrix(val_history_over_CV).mean(0)).reshape(-1))
    confusion_matrix_test_best = confusion_matrix_test_CV[0][best_epoch_id]
    for i in range(1, num_folds):
        confusion_matrix_test_best += confusion_matrix_test_CV[i][best_epoch_id]

    return train_history_over_CV, val_history_over_CV, confusion_matrix_test_best


def prep_train_validate_data_no_smote(data_dir, num_folds, fold_id, batch_size):
    data_loader = SeqDataLoader(data_dir, num_folds, fold_id, classes=5)

    X_train, y_train, X_test, y_test = data_loader.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0], 1)

    total_train_samples = X_train.shape[0]
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    classes_distribution = []
    char2numY = dict(zip(classes, range(len(classes))))
    for cl in classes:
        classes_distribution.append(len(np.where(y_train == char2numY[cl])[0]))

    # print(y_train.shape[0])
    # print(classes_distribution)
    temp = np.copy(classes_distribution)
    temp.sort()
    second_max = temp[-2]
    nb_samples_to_remove = classes_distribution[2] - second_max
    nb_samples_removed = 0
    indexes_to_remove = []

    for i in range(total_train_samples):
        if y_train[i] == 2 and nb_samples_removed < nb_samples_to_remove:
            indexes_to_remove.append(i)
            nb_samples_removed +=1

    # X_train = np.delete(X_train, indexes_to_remove, 0)
    # y_train = np.delete(y_train, indexes_to_remove, 0)


    # print(nb_samples_to_remove, len(indexes_to_remove))

    total_training_length = (X_train.shape[0] // batch_size) * batch_size
    total_test_length = (X_test.shape[0] // batch_size) * batch_size

    X_train = X_train[:total_training_length]
    y_train = y_train[:total_training_length]
    X_test = X_test[:total_test_length]
    y_test = y_test[:total_test_length]

    return X_train, y_train, X_test, y_test

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )


def prep_train_validate_data_CV(data_dir, fold_id, batch_size, seq_len=1):
    max_time_step = 10

    x = np.load(data_dir + "/trainData__SMOTE_all_10s_f" + str(fold_id) +".npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_dir + "/trainData__SMOTE_all_10s_f" + str(fold_id) +"_TEST.npz")
    X_test = x2["x"]
    y_test = x2["y"]

    X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]

    y_train = np.array(list(chain.from_iterable(zip(*repeat(y_train, seq_len)))))
    y_test = np.array(list(chain.from_iterable(zip(*repeat(y_test, seq_len)))))

    X_train = np.reshape(X_train, [-1, 1, 3000//seq_len])
    y_train = np.reshape(y_train, [-1, 1])
    X_test = np.reshape(X_test, [-1, 1, 3000//seq_len])
    y_test = np.reshape(y_test, [-1, 1])

    X_train = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    y_train = y_train[:(X_train.shape[0] // batch_size) * batch_size]
    X_test = X_test[:(X_test.shape[0] // batch_size) * batch_size, :]
    y_test = y_test[:(y_test.shape[0] // batch_size) * batch_size]

    return X_train, y_train, X_test, y_test

def prep_train_validate_data_CV_RUS(data_dir, fold_id, batch_size, seq_len=1):
    max_time_step = 10

    x = np.load(data_dir + "/trainData_fpz-cz_RUS_all_10s_f" + str(fold_id) +".npz")
    X_train = x["x"]
    y_train = x["y"]

    x2 = np.load(data_dir + "/trainData_fpz-cz_RUS_all_10s_f" + str(fold_id) +"_TEST.npz")
    X_test = x2["x"]
    y_test = x2["y"]

    X_train = X_train[:(X_train.shape[0] // max_time_step) * max_time_step, :]
    y_train = y_train[:(X_train.shape[0] // max_time_step) * max_time_step]

    # shuffle training data_2013
    permute = np.random.permutation(len(y_train))
    X_train = np.asarray(X_train)
    X_train = X_train[permute]
    y_train = y_train[permute]

    y_train = np.array(list(chain.from_iterable(zip(*repeat(y_train, seq_len)))))
    y_test = np.array(list(chain.from_iterable(zip(*repeat(y_test, seq_len)))))

    X_train = np.reshape(X_train, [-1, 1, 3000//seq_len])
    y_train = np.reshape(y_train, [-1, 1])
    X_test = np.reshape(X_test, [-1, 1, 3000//seq_len])
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
    print("Epoch:", np.argmax(test)+1, " max_acc=", max_val)
    xmax = np.argmax(test)
    ymax = test[xmax]


    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(xmax, ymax, 'bo', label="_nolegend_")

    axes = plt.gca()
    axes.set_ylim([0, 100])

    plt.plot(range(1, num_epoch + 1), train)
    plt.plot(range(1, num_epoch + 1), test)
    plt.legend(['Average Train Accuracy', 'Average Test Accuracy'], loc='lower right', prop={'size': 12})
    plt.suptitle('Cross Validation Performance', fontsize=15.0, y=1.08, fontweight='bold')
    plt.title('Best accuracy: ' + '{0:.2f}'.format(max_val) + ' % (epoch: ' + str(np.argmax(test)+1) + ')', fontsize=13.0,
              y=1.08, fontweight='bold')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch')
    plt.show()


def plot_one_validation_history(mlp, conv, convBatch, convLSTM):
    num_epoch = len(mlp)

    mlp = np.asarray(np.matrix(mlp).mean(0)).reshape(-1)
    conv = np.asarray(np.matrix(conv).mean(0)).reshape(-1)
    convBatch = np.asarray(np.matrix(convBatch).mean(0)).reshape(-1)
    convLSTM = np.asarray(np.matrix(convLSTM).mean(0)).reshape(-1)

    xmax = np.argmax(mlp)
    ymax = mlp[xmax]


    # print(max_val, ymax, xmax)

    # best_epoch_mlp = np.argmax(mlp_mean)

    # print(num_epoch)
    plt.figure(figsize=(16, 6))
    plt.plot(np.argmax(mlp), mlp[np.argmax(mlp)], 'bo', label="_nolegend_")
    plt.plot(np.argmax(conv), conv[np.argmax(conv)], 'yo', label="_nolegend_")
    plt.plot(np.argmax(convBatch), convBatch[np.argmax(convBatch)], 'go', label="_nolegend_")
    plt.plot(np.argmax(convLSTM), convLSTM[np.argmax(convLSTM)], 'ro', label="_nolegend_")

    axes = plt.gca()
    axes.set_ylim([0, 100])

    plt.plot(range(1, num_epoch + 1), mlp)
    plt.plot(range(1, num_epoch + 1), conv)
    plt.plot(range(1, num_epoch + 1), convBatch)
    plt.plot(range(1, num_epoch + 1), convLSTM)
    plt.legend(['MLP', 'Conv1D', 'ConvBatchNorm', 'ConvLSTM'], loc='lower right', prop={'size': 12})

    plt.title('Models Accuracy for TestSet')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch')
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


def print_confusion_matrix_accuracy(confusion_matrix_test_best, classes):
    confusion_matrix_accuracy = (confusion_matrix_test_best.diag().numpy() / confusion_matrix_test_best.sum(
        1).numpy()) * 100
    for i, cl in enumerate(classes):
        print("Acc ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')


def get_curr_time():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S")




EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)

class EDFEndOfData(Exception): pass


def tal(tal_str):
  '''Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  '''
  exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
    '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
    '(\x14(?P<annotation>[^\x00]*))?' + \
    '(?:\x14\x00)'

  def annotation_to_list(annotation):
    return str(annotation.encode('utf-8')).split('\x14') if annotation else []

  def parse(dic):
    return (
      float(dic['onset']),
      float(dic['duration']) if dic['duration'] else 0.,
      annotation_to_list(dic['annotation']))

  return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


def edf_header(f):
  h = {}
  assert f.tell() == 0  # check file position
  assert f.read(8) == '0       '

  # recording info)
  h['local_subject_id'] = f.read(80).strip()
  h['local_recording_id'] = f.read(80).strip()

  # parse timestamp
  (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
  (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
  h['date_time'] = str(datetime.datetime(year + 2000, month, day,
    hour, minute, sec))

  # misc
  header_nbytes = int(f.read(8))
  subtype = f.read(44)[:5]
  h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
  h['contiguous'] = subtype != 'EDF+D'
  h['n_records'] = int(f.read(8))
  h['record_length'] = float(f.read(8))  # in seconds
  nchannels = h['n_channels'] = int(f.read(4))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).strip() for n in channels]
  h['transducer_type'] = [f.read(80).strip() for n in channels]
  h['units'] = [f.read(8).strip() for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['prefiltering'] = [f.read(80).strip() for n in channels]
  h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
  f.read(32 * nchannels)  # reserved

  #assert f.tell() == header_nbytes
  return h


class BaseEDFReader:
  def __init__(self, file):
    self.file = file


  def read_header(self):
    self.header = h = edf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    assert np.all(phys_range > 0)
    assert np.all(dig_range > 0)
    self.gain = phys_range / dig_range


  def read_raw_record(self):
    '''Read a record with data_2013 and return a list containing arrays with raw
    bytes.
    '''
    result = []
    for nsamp in self.header['n_samples_per_record']:
      samples = self.file.read(nsamp * 2)
      if len(samples) != nsamp * 2:
        raise EDFEndOfData
      result.append(samples)
    return result


  def convert_record(self, raw_record):
    '''Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    '''
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
      if h['label'][i] == EVENT_CHANNEL:
        ann = tal(samples)
        time = ann[0][0]
        events.extend(ann[1:])
        # print(i, samples)
        # exit()
      else:
        # 2-byte little-endian integers
        dig = np.fromstring(samples, '<i2').astype(np.float32)
        phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
        signals.append(phys)

    return time, signals, events


  def read_record(self):
    return self.convert_record(self.read_raw_record())


  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    except EDFEndOfData:
      pass


def load_edf(edffile):
  if isinstance(edffile, basestring):
    with open(edffile, 'rb') as f:
      return load_edf(f)  # convert filename to file

  reader = BaseEDFReader(edffile)
  reader.read_header()

  h = reader.header
  log.debug('EDF header: %s' % h)

  # get sample rate info
  nsamp = np.unique(
    [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
    if l != EVENT_CHANNEL])
  assert nsamp.size == 1, 'Multiple sample rates not supported!'
  sample_rate = float(nsamp[0]) / h['record_length']

  rectime, X, annotations = zip(*reader.records())
  X = np.hstack(X)
  annotations = reduce(operator.add, annotations)
  chan_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]

  # create timestamps
  if reader.header['contiguous']:
    time = np.arange(X.shape[1]) / sample_rate
  else:
    reclen = reader.header['record_length']
    within_rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
    time = np.hstack([t + within_rec_time for t in rectime])

  tup = namedtuple('EDF', 'X sample_rate chan_lab time annotations')
  return tup(X, sample_rate, chan_lab, time, annotations)