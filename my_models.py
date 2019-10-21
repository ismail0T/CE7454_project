import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTM(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTM, self).__init__()
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





        # LL2:   128  -->  classes
        # x = self.linear2(x)

        # return x


class ConvSimple(nn.Module):

    def __init__(self):
        super(ConvSimple, self).__init__()
        self.n_classes = 5
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
