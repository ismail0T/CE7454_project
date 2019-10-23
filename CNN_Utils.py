import torch
import torch.nn as nn
import time
import numpy as np


class CNN_Utils():
    def __init__(self, my_lr, batch_size, epochs):
        self.learning_rate = my_lr
        self.batch_size = batch_size
        self.epochs = epochs

    def test_model_conv_lstm(self, net, test_x, test_y, bi_dir, device):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            num_lstm_layers = 1
            hidden_dim_of_lstm1 = 256
            confusion_matrix = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, len(test_x), self.batch_size):
                test_x_batch = test_x[i:i + self.batch_size]
                test_y_batch = test_y[i:i + self.batch_size]

                test_x_batch = torch.from_numpy(test_x_batch).to(device)
                test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

                # initial hidden states
                h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1,
                                hidden_dim_of_lstm1)  ### if bi_dir=True, first dimension is 2 for bi-directional
                c = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1, hidden_dim_of_lstm1)

                h = h.to(device)
                c = c.to(device)

                m_input = test_x_batch
                outputs, h, c = net(m_input.float(), h, c)

                outputs = outputs.view(test_x_batch.shape[0], 5)
                _, predicted = torch.max(outputs.data, 1)
                total += test_y_batch.size(0)
                correct += (predicted == test_y_batch.long()).sum().item()

                # confusion matrix
                stacked = torch.stack(
                    (
                        test_y_batch.int()
                        , predicted.int()
                    )
                    , dim=1
                )
                for p in stacked:
                    tl, pl = p.tolist()
                    confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

        acc_test = (correct / total) * 100

        print('Test Accuracy: {} %'.format(acc_test))

        # Confusion matrix
        print("Test Confusion matrix ")
        classes = ['W', 'N1', 'N2', 'N3', 'REM']
        confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
        for i, cl in enumerate(classes):
            print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

        # Test the model
        return acc_test

    def train_model_conv_lstm(self, net, train_x, train_y, test_x, test_y, bi_dir, device):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        start = time.time()

        train_history = []
        test_history = []
        num_lstm_layers = 1
        hidden_dim_of_lstm1 = 256

        for epoch in range(1, self.epochs+1):
            net.train()
            if not epoch % 10:
                lr = lr / 1.5

            optimizer = torch.optim.SGD(net.parameters(), lr=lr)

            running_loss = 0
            loss_list = []
            acc_list = []
            correct_all = 0
            total_all = 0
            confusion_matrix = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                # initial hidden states
                h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1,
                                hidden_dim_of_lstm1)  # if bi_dir=True, first dimension is 2 for bi-directional
                c = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1, hidden_dim_of_lstm1)

                h = h.to(device)
                c = c.to(device)

                optimizer.zero_grad()
                outputs, h, c = net(train_x_batch.float(), h, c)

                outputs = outputs.view(train_x_batch.shape[0], 5)
                loss = criterion(outputs, train_y_batch.long())

                # Backprop and perform SGD optimisation

                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = train_y_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == train_y_batch.long()).sum().item()

                # Statistics
                running_loss += loss.detach().item()
                loss_list.append(loss.item())
                acc_list.append(correct / total)
                correct_all += correct
                total_all += total

                # confusion matrix
                stacked = torch.stack(
                    (
                        train_y_batch.int()
                        , predicted.int()
                    )
                    , dim=1
                )
                for p in stacked:
                    tl, pl = p.tolist()
                    confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

            epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Confusion matrix
            print("Train Confusion matrix ")
            classes = ['W', 'N1', 'N2', 'N3', 'REM']
            confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
            for i, cl in enumerate(classes):
                print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

            # Test the model
            acc_test = self.test_model_conv_lstm(net, test_x, test_y, bi_dir, device)
            test_history.append(acc_test)
            print("\n")


        return train_history, test_history

    def test_model_cnn(self, net, test_x, test_y, device):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, len(test_x), self.batch_size):
                test_x_batch = test_x[i:i + self.batch_size]
                test_y_batch = test_y[i:i + self.batch_size]

                test_x_batch = torch.from_numpy(test_x_batch).to(device)
                test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

                m_input = test_x_batch
                outputs = net(m_input.float())
                _, predicted = torch.max(outputs.data, 1)
                total += test_y_batch.size(0)
                correct += (predicted == test_y_batch.long()).sum().item()

                # confusion matrix
                stacked = torch.stack(
                    (
                        test_y_batch.int()
                        , predicted.int()
                    )
                    , dim=1
                )

                for p in stacked:
                    tl, pl = p.tolist()
                    confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

        acc_test = (correct / total) * 100
        print('Test Accuracy: {} %'.format(acc_test))

        # Confusion matrix
        print("Test Confusion matrix ")
        classes = ['W', 'N1', 'N2', 'N3', 'REM']
        confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
        for i, cl in enumerate(classes):
            print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

        return acc_test

    def train_model_cnn(self, net, train_x, train_y, test_x, test_y, device):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        start = time.time()

        train_history = []
        test_history = []

        for epoch in range(1, self.epochs+1):

            if not epoch % 10:
                lr = lr / 1.0

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            running_loss = 0
            loss_list = []
            acc_list = []
            correct_all = 0
            total_all = 0
            confusion_matrix = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                outputs = net(train_x_batch.float())
                loss = criterion(outputs, train_y_batch.long())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = train_y_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == train_y_batch.long()).sum().item()

                # Statistics
                running_loss += loss.detach().item()
                loss_list.append(loss.item())
                acc_list.append(correct / total)
                correct_all += correct
                total_all += total

                # confusion matrix
                stacked = torch.stack(
                    (
                        train_y_batch.int()
                        , predicted.int()
                    )
                    , dim=1
                )
                for p in stacked:
                    tl, pl = p.tolist()
                    confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

            epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Confusion matrix
            print("Train Confusion matrix ")
            classes = ['W', 'N1', 'N2', 'N3', 'REM']
            confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
            for i, cl in enumerate(classes):
                print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

            # Test the model
            acc_test = self.test_model_cnn(net, test_x, test_y, device)
            test_history.append(acc_test)
            print("\n")
            # print('Test Accuracy: {} %'.format(acc_test)+"\n")

        return train_history, test_history
