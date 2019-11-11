import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

from NN_Utils import *
from my_models import *
from dataloader import SeqDataLoader
# from Utils_Old import *
import time, sys
import copy
import time
import math

from old_contrib.Utils_Old import prep_train_validate_data_CV_seq2seq


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



data_dir = "../../data_2013/traindata_eeg"

classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)

num_epochs = 15
batch_size = 128
learning_rate = 0.001

device = torch.device("cuda:1")

# def run_experiment_cross_validation():
#     CNNutils = CNN_Utils(learning_rate, batch_size, num_epochs)
#     train_history_over_CV = []
#     val_history_over_CV = []
#     confusion_matrix_train_CV = []
#     confusion_matrix_test_CV = []
#     num_folds = 3
#
#     print('num_folds: ', num_folds, ' num_epochs: ', num_epochs)
#
#
#     for fold_id in range(0, num_folds):
#         # Loading Data
#         X_train, y_train, X_test, y_test = prep_train_validate_data_CV_seq2seq(data_dir, fold_id, batch_size, 10, classes)
#         if fold_id == 0:
#             print('X_train Data Shape: ', X_train.shape, '  y_train Data Shape: ', y_train.shape)
#             print('X_test Data Shape:  ', X_test.shape, '   y_test Data Shape:  ', y_test.shape)
#
#             print('\n')
#             # X_train Data Shape:  (5632, 10, 3000)   y_train Data Shape:  (5632, 12)
#             # X_test Data Shape:   (2068, 10, 3000)    y_test Data Shape:   (2068, 12)
#         # sys.exit()
#
#
#
#
#         char2numY = dict(zip(classes, range(len(classes))))
#         # for cl in classes:
#         #     print("__Train ", cl, len(np.where(y_train == char2numY[cl])[0]), " => ",
#         #           len(np.where(y_test == char2numY[cl])[0]))
#
#         print("\nFold <" + str(fold_id + 1) + ">")
#         # Train Data Shape:  (38912, 1, 3000)   Test Data Shape:  (2048, 1, 3000)
#         # Train  W 7305  =>  639
#         # Train  N1 2651  =>  74
#         # Train  N2 16375  =>  860
#         # Train  N3 5270  =>  263
#         # Train  REM 7311  =>  212
#
#         # sys.exit()
#         # model #
#         net = Seq2Seq11()
#         print("Seq2Seq..")
#         if fold_id == 0:
#             display_num_param(net)
#         net = net.to(device)
#
#         train_history, validation_history, confusion_matrix_train_list, confusion_matrix_test_list = CNNutils.train_model_conv_lstm(net, X_train, y_train, X_test, y_test, False, device)
#         # print('train_history', train_history)
#         # accumulate history for each CV loop, then take average
#         train_history_over_CV.append(train_history)
#         val_history_over_CV.append(validation_history)
#         confusion_matrix_train_CV.append(confusion_matrix_train_list)
#         confusion_matrix_test_CV.append(confusion_matrix_test_list)
#         # print(confusion_matrix_test)
#
#         del net
#
#     print(train_history_over_CV)
#     print(val_history_over_CV)
#
#     confusion_matrix_test_CV_final = confusion_matrix_test_CV[0]
#     for i in range(1, num_folds):
#         confusion_matrix_test_CV_final += confusion_matrix_test_CV[i]
#
#     best_epoch_id = np.argmax(np.asarray(np.matrix(val_history_over_CV).mean(0)).reshape(-1))
#
#     confusion_matrix_test_best = confusion_matrix_test_CV_final[best_epoch_id]
#
#     confusion_matrix_accuracy = (confusion_matrix_test_best.diag().numpy() / confusion_matrix_test_best.sum(1).numpy()) * 100
#     for i, cl in enumerate(classes):
#         print("Acc ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')
#
#     plot_CV_history(train_history_over_CV, val_history_over_CV)
#     plot_confusion_matrix(confusion_matrix_test_best.data.numpy(), classes)
# plot_confusion_matrix(cm=tt.data.numpy(), target_names=classes)


# tt = torch.from_numpy(np.asarray(
#         [[119,  20,  85,  10,  37],
#         [ 46,  260,  44,   2,  33],
#         [13,  37, 300, 120,  79],
#         [ 21,   1,  88, 183,   1],
#         [18,  58, 134,   8, 235]]))

# plot_confusion_matrix(tt, classes)
# run_experiment_cross_validation()
# ttttt()

# num_epochs = 25
# batch_size = 64
# learning_rate = 0.04

# def run_experiment_simple_validation000():
#     print(' num_epochs: ', num_epochs)
#     CNNutils = CNN_Utils(learning_rate, batch_size, num_epochs)
#     X_train, y_train, X_test, y_test = prep_train_validate_data(data_dir, batch_size)
#     print('Train Data Shape: ', X_train.shape, '  Test Data Shape: ', X_test.shape)
#     print('\n')
#
#     # model #
#     bi_dir = True
#     # net = ConvLSTM(bi_dir)
#     enc = Encoder()
#     dec = Decoder()
#     net = Seq2Seq(enc, dec, device)
#     display_num_param(net)
#     net = net.to(device)
#
#     train_history, validation_history = CNNutils.train_model_conv_lstm(net, X_train, y_train, X_test, y_test, bi_dir, device)
#     print('train_history', train_history)
#     print('validation_history', validation_history)
#
#     del net
#
#     plot_one_validation_history(train_history, validation_history)


# run_experiment_simple_validation000()



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(3000, 256)

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(256, hidden_size)

    def forward(self, input, hidden):

        # print('000 input_tensor.shape', input.shape)
        # print('111 input_tensor.shape', input.view(1, 1, -1).shape)

        # embedded = self.embedding(input).view(1, 1, -1)
        x = self.linear1(input)
        x = F.relu(x)

        output = x.view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        print("AttnDecoderRNN 1 input", input.shape, input)
        print("AttnDecoderRNN 1 hidden", hidden.shape)
        print("AttnDecoderRNN 1 encoder_outputs", encoder_outputs.shape)

        embedded = input.view(1, 1, -1)
        # embedded = self.dropout(embedded)

        print('embedded', embedded.shape)
        print('embedded[0]', embedded[0])
        print('\nhidden', hidden.shape)
        # print('hidden[0]', hidden[0])

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        print("output 9", output.shape)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5

SOS_token = 5
EOS_token = 6


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    correct = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        # print('000 encoder_output.shape', encoder_output.shape)

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS_token=5=SOD

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # predicted_list = []
        # correct_list = []
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))

            # Track the accuracy
            total = target_tensor.size(0)
            _, predicted = torch.max(decoder_output.data, 1)
            correct += (predicted == target_tensor[di].long()).sum().item()

            # predicted_list.append(predicted)
            # correct_list.append(correct)



            decoder_input = target_tensor[di]  # Teacher forcing
        # print(target_tensor)
        # print(predicted_list)
        # print(correct)
        # sys.exit()

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, correct


X_train, y_train, X_test, y_test = prep_train_validate_data_CV_seq2seq(data_dir, 0, 1, 10, classes)
print('X_train Data Shape: ', X_train.shape, '  y_train Data Shape: ', y_train.shape)
print('X_test Data Shape:  ', X_test.shape, '   y_test Data Shape:  ', y_test.shape)
# import torchaudio


def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every


    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    #     print(len(pairs))
    #     print(len(training_pairs))

    for epoch in range(10):
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        correct_total = 0

        for iter in range(1, n_iters + 1):
            input_tensor = torch.Tensor(X_train[iter]).to(device)
            target_tensor = torch.Tensor(y_train[iter]).long().to(device)

            loss, correct = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            correct_total += correct
            # sys.exit()

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
                print(correct, '    ', correct_total, correct_total/(n_iters*10)*100, '%', '\n')

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


hidden_size = 256
encoder1 = EncoderRNN(7, hidden_size).to(device)
attn_decoder1 = DecoderRNN(hidden_size, 7).to(device)


trainIters(encoder1, attn_decoder1, 5600, print_every=500)





