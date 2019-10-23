import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.linear1 = nn.Linear(6016, 128)

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


class ConvLSTM(nn.Module):

    def __init__(self, bi_dir):
        super(ConvLSTM, self).__init__()
        self.n_classes = 5
        self.hidden_dim = 256
        self.bi_dir = bi_dir

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

class ConvSimpleBest(nn.Module):

    def __init__(self):
        super(ConvSimpleBest, self).__init__()
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
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)

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
