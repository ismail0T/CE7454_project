from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Utils import *
path_to_saved_models = "saved_models"
path_to_test_results = "test_results"


def plot_test_sequence():
    conv_net = torch.load(os.path.join(path_to_saved_models, "ConvSimple_X_TSNE.pt"))
    conv_net_y = torch.load(os.path.join(path_to_saved_models, "ConvSimple_Y_TSNE.pt"))

    for k in [0, 1, 3, 7, 15, 24]:
        conv_tsne = TSNE(n_components=2, random_state=1).fit_transform(
            (Variable(conv_net[k]).data).cpu().numpy().reshape(2048, -1).astype(np.float64))
        yy = Variable(conv_net_y[k]).data.cpu().numpy().astype(np.float64)
        fig, ax = plt.subplots(figsize=(8, 5))

        scatter = ax.scatter(conv_tsne[:, 0], conv_tsne[:, 1], c=yy)
        ax.text(0.95, 0.05, "Epoch " + str(k), transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', horizontalalignment='right')
        legend1 = ax.legend(*scatter.legend_elements(),
                                loc="lower left", title="Classes")
        ax.add_artist(legend1)
        # plt.savefig(os.path.join(path_to_tsne_figures, str(k)+".png"))
        plt.show()

def get_last_layer_output_convLSTM(net, X_test, y_test, device):
    batch_size = 1
    bi_dir = False
    test_x, test_y = X_test, y_test
    net.eval()
    conv_outputs = torch.zeros((test_x.shape[0], 256))
    with torch.no_grad():
        num_lstm_layers = 1
        hidden_dim_of_lstm1 = 256
        for i in range(0, len(test_x)):
            test_x_batch = test_x[i:i + batch_size]
            test_y_batch = test_y[i:i + batch_size]

            test_x_batch = torch.from_numpy(test_x_batch).to(device)
            test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

            # initial hidden states
            h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1,
                            hidden_dim_of_lstm1)
            c = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1, hidden_dim_of_lstm1)

            h = h.to(device)
            c = c.to(device)

            m_input = test_x_batch
            outputs, lstm_out = net(m_input.float(), h, c)
            conv_outputs[i, :] = lstm_out.squeeze()

    return conv_outputs


def get_last_layer_output_conv1D(net, X_test, y_test, device):
    batch_size = 1
    test_x, test_y = X_test, y_test
    net.eval()
    conv_output = torch.zeros((test_x.shape[0], 128))
    with torch.no_grad():
        for i in range(0, len(test_x)):
            test_x_batch = test_x[i:i + batch_size]
            test_y_batch = test_y[i:i + batch_size]

            test_x_batch = torch.from_numpy(test_x_batch).to(device)
            test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

            m_input = test_x_batch
            line2, lin1 = net(m_input.float())
            conv_output[i, :] = lin1

    return conv_output


def get_last_layer_output_mlp(net, X_test, y_test, device):
    batch_size = 1
    test_x, test_y = X_test, y_test
    net.eval()
    mlp_output = torch.zeros((test_x.shape[0], 128))
    with torch.no_grad():
        for i in range(0, len(test_x)):  # , batch_size):
            test_x_batch = test_x[i:i + batch_size]
            test_y_batch = test_y[i:i + batch_size]

            test_x_batch = torch.from_numpy(test_x_batch).to(device)
            test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

            m_input = test_x_batch
            scores, y = net(m_input.float())
            mlp_output[i, :] = y

    return mlp_output


def plot_model_last_layer_after_training(mlp_output, X_test, y_test):
    mlp_tsne = TSNE(n_components=2,random_state=1).fit_transform((Variable(mlp_output).data).cpu().numpy().reshape(X_test.shape[0],-1).astype(np.float64))

    fig, ax = plt.subplots(figsize=(16, 10))

    scatter = ax.scatter(mlp_tsne[:, 0], mlp_tsne[:, 1], c=y_test.squeeze())
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()


def plot_model_last_layer(X_test, y_test):
    X_tsne = TSNE(n_components=2).fit_transform(X_test.reshape(X_test.shape[0], -1))
    fig, ax = plt.subplots(figsize=(16, 10))

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test.squeeze())
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.n_classes = 5

        self.layer1 = nn.Linear(3000, 1024, bias=False)
        self.layer2 = nn.Linear(1024, 512, bias=False)
        self.layer3 = nn.Linear(512, 128, bias=False)
        self.layer4 = nn.Linear(128, self.n_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1) * x.size(2))

        y = self.layer1(x)
        y = F.relu(y)

        y = self.layer2(y)
        y = F.relu(y)

        y = self.layer3(y)
        y = F.relu(y)

        scores = self.layer4(y)

        return scores, y


class ConvSimple(nn.Module):
    def __init__(self):
        super(ConvSimple, self).__init__()
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
        xx = self.dropout1(x)

        # LL2:   128  -->  classes
        x = self.linear2(xx)

        return x,xx

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
        self.dropout2 = nn.Dropout(0.6)

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
        g_seq = cnn_x.unsqueeze(dim=1)
        lstm_out, (h_final, c_final) = self.lstm(g_seq, (h_init, c_init))

        # Droput
        lstm_out = self.dropout2(lstm_out)

        # linear
        cnn_lstm_out = self.hidden2label1(lstm_out)  # activations are implicit

        # output
        scores = cnn_lstm_out

#         return scores, h_final, c_final
        return scores, lstm_out




