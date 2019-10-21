import torch
import torch.nn as nn
import torch.nn.functional as F


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
