import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn import TemporalConvNet


class ConvLSTM01(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTM01, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir

        # CL1:   28 x 28  -->    64 x 3'000
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, padding=1, stride=2)

        # MP1: 64 x 3'000 -->    64 x 1'500
        self.pool1 = nn.MaxPool1d(2, stride=4)

        # CL2:   64 x 1'500  -->    64 x 1'500
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # MP2: 64 x 1'500  -->    64 x 750
        self.pool2 = nn.MaxPool1d(2)

        # CL3:   64 x 750  -->    64 x 750
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.to_pad = 1
        # MP3: 64 x 750  -->    64 x 375
        self.pool3 = nn.MaxPool1d(2, padding=self.to_pad)

        self.linear1 = nn.Linear(6016, 256)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 256
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    def forward(self, x, h_init, c_init):
        # print(x.shape)
        # CL1:   1 x 3'000  -->    64 x 3'000
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)

        # MP1: 64 x 3'000 -->    64 x 1'500
        x = self.pool1(x)
        # print(x.shape)

        x = self.dropout1(x)

        # CL2:   64 x 1'500  -->    64 x 1'500
        x = self.conv2(x)
        x = F.relu(x)
        # print(x.shape)

        # MP2: 64 x 1'500  -->    64 x 750
        x = self.pool2(x)
        # print(x.shape)

        # CL3:   64 x 750  -->    64 x 750
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.dropout1(x)

        # MP3: 64 x 376 = 24'064
        x = self.pool3(x)
        # print(x.shape)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)



        cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = cnn_x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout1(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

        return scores, h_final, c_final

class ConvLSTM00(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTM00, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir

        self.conv1 = nn.Conv1d(1, 64, kernel_size=50, padding=1, stride=6)
        self.pool1 = nn.MaxPool1d(1, stride=8)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=1, stride=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=8, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, padding=1, stride=1)
        self.pool2 = nn.MaxPool1d(1, stride=4)

        self.conv1_2 = nn.Conv1d(1, 64, kernel_size=400, padding=1, stride=50)
        self.pool1_2 = nn.MaxPool1d(1, stride=4)
        self.dropout1_2 = nn.Dropout(0.5)

        self.conv2_2 = nn.Conv1d(64, 128, kernel_size=6, padding=1, stride=1)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=6, padding=1, stride=1)
        self.conv4_2 = nn.Conv1d(128, 128, kernel_size=6, padding=1, stride=1)
        self.pool2_2 = nn.MaxPool1d(1, stride=2)

        self.linear1 = nn.Linear(1920, 128)

        # self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 1920
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    def forward(self, x, h_init, c_init):
        out_time = self.conv1(x)
        out_time = F.relu(out_time)
        out_time = self.pool1(out_time)
        out_time = self.dropout1(out_time)
        out_time = self.conv2(out_time)
        out_time = F.relu(out_time)
        out_time = self.conv3(out_time)
        out_time = F.relu(out_time)
        out_time = self.conv4(out_time)
        out_time = F.relu(out_time)
        out_time = self.pool2(out_time)

        out_freq = self.conv1_2(x)
        out_freq = F.relu(out_freq)
        out_freq = self.pool1_2(out_freq)
        out_freq = self.dropout1_2(out_freq)
        out_freq = self.conv2_2(out_freq)
        out_freq = F.relu(out_freq)
        out_freq = self.conv3_2(out_freq)
        out_freq = F.relu(out_freq)
        out_freq = self.conv4_2(out_freq)
        out_freq = F.relu(out_freq)
        out_freq = self.pool2_2(out_freq)

        x = torch.cat((out_time, out_freq), 2)
        x = self.dropout1(x)


        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        # x = self.linear1(x)
        # x = F.relu(x)
        #
        # # Droput
        # x = self.dropout1(x)

        cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = cnn_x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout2(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

        return scores, h_final, c_final

class Seq2Seq11(nn.Module):

    def __init__(self):
        super(Seq2Seq11, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256

        # self.conv1 = nn.Conv1d(1, 32, kernel_size=10, padding=1, stride=2)
        # self.conv2 = nn.Conv1d(32, 32, kernel_size=10, padding=1, stride=2)
        # self.pool1 = nn.MaxPool1d(2, stride=4)
        #
        # self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(2, stride=2)
        #
        # # self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # # self.pool3 = nn.MaxPool1d(2, stride=1)
        #
        # self.linear1 = nn.Linear(576, 128)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 3000
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, h_init, c_init):

        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool1(x)
        # x = self.dropout1(x)
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool2(x)
        # x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        # x = self.linear1(x)
        # x = F.relu(x)
        #
        # # Droput
        # x = self.dropout1(x)



        # cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = x #.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout1(lstm_out)

        # linear
        lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        lstm_out = self.dropout1(lstm_out)

        # output
        scores = lstm_out

        return scores, h_final, c_final


class MyLSTM(nn.Module):

    def __init__(self, bi_dir):
        super(MyLSTM, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir

        self.linear1 = nn.Linear(3000, 256)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 256
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    def forward(self, x, h_init, c_init):
        x = self.linear1(x)
        x = F.relu(x)
        # x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))

        # LSTM
        g_seq = x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout1(lstm_out)

        # linear
        lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        lstm_out = self.dropout1(lstm_out)

        # output
        scores = lstm_out

        return scores, h_final, c_final


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.n_classes = 5

        self.layer1 = nn.Linear(3000, 256, bias=False)
        self.layer2 = nn.Linear(256, self.n_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1) * x.size(2))

        y = self.layer1(x)
        y = F.relu(y)

        scores = self.layer2(y)

        return  scores

class ConvGRU(nn.Module):

    def __init__(self, bi_dir):
        super(ConvGRU, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir
        self.rnn_type = 'gru'
        self.num_layers = 1

        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, padding=1, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=10, padding=1, stride=3)
        self.pool1 = nn.MaxPool1d(2, stride=6)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        # self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool1d(2, stride=1)

        self.linear1 = nn.Linear(1728, 128)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 128
        self.gru = nn.GRU(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    def init_hidden(self, batch_size):
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers, 1, self.hidden_dim)
        elif self.rnn_type == 'lstm':
            return (
                torch.zeros(self.num_layers * (1 + int(self.bi_dir)), batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * (1 + int(self.bi_dir)), batch_size, self.hidden_dim))

    def forward(self, x, gru_hidden):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)



        cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = cnn_x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, gru_hidden = self.gru(g_seq, gru_hidden)

        # Droput
        lstm_out = self.dropout2(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

        return scores, gru_hidden





        # LL2:   128  -->  classes
        # x = self.linear2(x)

        # return x



class ConvLSTM(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTM, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir
        self.num_layers = 1
        self.rnn_type = 'lstm'

        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, padding=1, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=10, padding=1, stride=3)
        self.pool1 = nn.MaxPool1d(2, stride=6)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=2)

        # self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool1d(2, stride=1)

        self.linear1 = nn.Linear(1728, 128)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 128
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    # def init_hidden(self, batch_size):
    #     if self.rnn_type == 'gru':
    #         return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
    #     elif self.rnn_type == 'lstm':
    #         return (
    #             torch.zeros(self.num_layers * (1 + int(self.bi_dir)), batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers * (1 + int(self.bi_dir)), batch_size, self.hidden_dim))

    def forward(self, x, h_init, c_init):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)



        cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = cnn_x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout2(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

        return scores, h_final, c_final





        # LL2:   128  -->  classes
        # x = self.linear2(x)

        # return x


class ConvLSTMOld(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTMOld, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir

        # CL1:   28 x 28  -->    64 x 3'000
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)

        # MP1: 64 x 3'000 -->    64 x 1'500
        self.pool1 = nn.MaxPool1d(2)

        # CL2:   64 x 1'500  -->    64 x 1'500
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # MP2: 64 x 1'500  -->    64 x 750
        self.pool2 = nn.MaxPool1d(2)

        # CL3:   64 x 750  -->    64 x 750
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.to_pad = 1
        # MP3: 64 x 750  -->    64 x 375
        self.pool3 = nn.MaxPool1d(2, padding=self.to_pad)

        self.linear1 = nn.Linear(24064, 128)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)

        # LL2:   128  -->  classes
        # self.linear2 = nn.Linear(128, self.n_classes)

        # LSTM
        self.lstm_in_dim = 128
        self.lstm = nn.LSTM(self.lstm_in_dim, self.hidden_dim, bidirectional=self.bi_dir)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim * (1 + int(self.bi_dir)), self.n_classes)

    def forward(self, x, h_init, c_init):
        # print(x.shape)
        # CL1:   1 x 3'000  -->    64 x 3'000
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)

        # MP1: 64 x 3'000 -->    64 x 1'500
        x = self.pool1(x)
        # print(x.shape)

        x = self.dropout1(x)

        # CL2:   64 x 1'500  -->    64 x 1'500
        x = self.conv2(x)
        x = F.relu(x)
        # print(x.shape)

        # MP2: 64 x 1'500  -->    64 x 750
        x = self.pool2(x)
        # print(x.shape)

        # CL3:   64 x 750  -->    64 x 750
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.dropout1(x)

        # MP3: 64 x 376 = 24'064
        x = self.pool3(x)
        # print(x.shape)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)



        cnn_x = F.relu(x)
        # print('cnn_x', cnn_x.shape)
        # LSTM
        g_seq = cnn_x.unsqueeze(dim=1)
        # print('g_seq', g_seq.shape)

        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout1(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

        return scores, h_final, c_final


class ConvSimple(nn.Module):
    def __init__(self):
        super(ConvSimple, self).__init__()
        self.n_classes = 5
        # CL1:   28 x 28  -->    64 x 3'000
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, padding=1, stride=2)

        # MP1: 64 x 3'000 -->    64 x 1'500
        self.pool1 = nn.MaxPool1d(2, stride=4)

        # CL2:   64 x 1'500  -->    64 x 1'500
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # MP2: 64 x 1'500  -->    64 x 750
        self.pool2 = nn.MaxPool1d(2)

        # CL3:   64 x 750  -->    64 x 750
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # MP3: 64 x 750  -->    64 x 375
        self.pool3 = nn.MaxPool1d(2, padding=1)

        self.linear1 = nn.Linear(6016, 128)

        self.dropout1 = nn.Dropout(0.5)

        # LL2:   128  -->  classes
        self.linear2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        # print(x.shape)
        # CL1:   1 x 3'000  -->    64 x 3'000
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)

        # MP1: 64 x 3'000 -->    64 x 1'500
        x = self.pool1(x)
        # print(x.shape)

        x = self.dropout1(x)

        # CL2:   64 x 1'500  -->    64 x 1'500
        x = self.conv2(x)
        x = F.relu(x)
        # print(x.shape)

        # MP2: 64 x 1'500  -->    64 x 750
        x = self.pool2(x)
        # print(x.shape)

        # CL3:   64 x 750  -->    64 x 750
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.dropout1(x)

        # MP3: 64 x 376 = 24'064
        x = self.pool3(x)
        # print(x.shape)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)

        # LL2:   128  -->  classes
        x = self.linear2(x)

        return x


class ConvSimpleSOTA(nn.Module):

    def __init__(self):
        super(ConvSimpleSOTA, self).__init__()
        self.n_classes = 5
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, padding=1, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=10, padding=1, stride=3)
        self.pool1 = nn.AvgPool1d(2, stride=6)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool1d(2, stride=2)

        self.conv5 = nn.Conv1d(64, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool_avg = nn.AvgPool1d(2)

        self.linear1 = nn.Linear(3328, 128)

        self.dropout1 = nn.Dropout(0.2)

        # LL2:   128  -->  classes
        self.linear2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool_avg(x)
        x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)

        # LL2:   128  -->  classes
        x = self.linear2(x)

        return x


class ConvSimpleBest(nn.Module):

    def __init__(self):
        super(ConvSimpleBest, self).__init__()
        self.n_classes = 5
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, padding=1, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=10, padding=1, stride=3)
        self.pool1 = nn.AvgPool1d(2, stride=6)
        self.bn1 = nn.BatchNorm1d(num_features=32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool1d(2, stride=2)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv5 = nn.Conv1d(64, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool_avg = nn.AvgPool1d(2)
        self.bn3 = nn.BatchNorm1d(num_features=256)


        self.linear1 = nn.Linear(3328, 128)

        self.dropout1 = nn.Dropout(0.02)

        # LL2:   128  -->  classes
        self.linear2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x = self.dropout1(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool2(x)
        # x = self.dropout1(x)

        x = self.conv5(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool_avg(x)
        # x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)

        # LL2:   128  -->  classes
        x = self.linear2(x)

        return x



class TCN00(nn.Module):
    def __init__(self):
        super(TCN00, self).__init__()
        input_size = 1
        output_size = 5
        num_channels = [16]*4
        kernel_size = 10
        dropout = 0.2

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x).transpose(1, 2)
        output = self.linear(output).double()
        return output # self.sig(output)

class TempConv(nn.Module):

    def __init__(self):
        super(TempConv, self).__init__()
        self.n_classes = 5
        self.conv1 = TemporalConvNet(1, 32, kernel_size=10)
        self.conv2 = TemporalConvNet(32, 32, kernel_size=10)
        self.pool1 = nn.AvgPool1d(2, stride=6)

        self.conv3 = TemporalConvNet(32, 64, kernel_size=3)
        self.conv4 = TemporalConvNet(64, 64, kernel_size=3)
        self.pool2 = nn.AvgPool1d(2, stride=2)

        self.conv5 = TemporalConvNet(64, 256, kernel_size=3)
        self.conv6 = TemporalConvNet(256, 256, kernel_size=3)
        self.pool_avg = nn.AvgPool1d(2)

        self.linear1 = nn.Linear(3328, 128)

        self.dropout1 = nn.Dropout(0.02)

        # LL2:   128  -->  classes
        self.linear2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        # x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x = self.dropout1(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool2(x)
        # x = self.dropout1(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool_avg(x)
        # x = self.dropout1(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        # print(x.shape)  # 24'064

        x = self.linear1(x)
        x = F.relu(x)

        # Droput
        x = self.dropout1(x)

        # LL2:   128  -->  classes
        x = self.linear2(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 3000
        self.hid_dim = 64
        self.n_layers = 1
        self.dropout = 0.3
        self.rnn = nn.LSTM(self.input_dim, self.hid_dim, self.n_layers, dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src):

        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.output_dim = 5
        self.hid_dim = 64
        self.n_layers = 1
        self.dropout = 0.5

        self.rnn = nn.LSTM(self.output_dim, self.hid_dim, self.n_layers, dropout=self.dropout)

        self.out = nn.Linear(self.hid_dim, self.output_dim)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        # embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs