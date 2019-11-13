import torch
import torch.nn as nn
import time


class NN_Utils():
    def __init__(self, my_lr, batch_size, epochs):
        self.learning_rate = my_lr
        self.batch_size = batch_size
        self.epochs = epochs

    def test_model_conv_gru(self, net, test_x, test_y, bi_dir, device, confusion_matrix_test):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            num_lstm_layers = 1
            hidden_dim_of_lstm1 = 256

            for i in range(0, len(test_x), self.batch_size):
                test_x_batch = test_x[i:i + self.batch_size]
                test_y_batch = test_y[i:i + self.batch_size]

                test_x_batch = torch.from_numpy(test_x_batch).to(device)
                test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

                # initial hidden states
                h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1,
                                hidden_dim_of_lstm1)  ### if bi_dir=True, first dimension is 2 for bi-directional

                h = h.to(device)

                m_input = test_x_batch
                outputs, h = net(m_input.float(), h)

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
                    confusion_matrix_test[tl, pl] = confusion_matrix_test[tl, pl] + 1

        acc_test = (correct / total) * 100

        print('Test Accuracy: {} %'.format(acc_test))

        # Confusion matrix
        # print("Test Confusion matrix ")
        # classes = ['W', 'N1', 'N2', 'N3', 'REM']
        # confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
        # for i, cl in enumerate(classes):
        #     print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

        # Test the model
        return acc_test, confusion_matrix_test

    def train_model_conv_gru(self, net, train_x, train_y, test_x, test_y, bi_dir, device):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        start = time.time()

        train_history = []
        test_history = []
        num_lstm_layers = 1
        hidden_dim_of_lstm1 = 256

        confusion_matrix_train = torch.zeros(5, 5, dtype=torch.int32)
        confusion_matrix_test = torch.zeros(5, 5, dtype=torch.int32)

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

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                # print(train_x_batch.shape)
                # print(train_y_batch.shape)
                # sys.exit()

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                # initial hidden states
                h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1,
                                 hidden_dim_of_lstm1)  # if bi_dir=True, first dimension is 2 for bi-directional
                # c = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1, hidden_dim_of_lstm1)
                #
                # h = h.to(device)
                # c = c.to(device)

                optimizer.zero_grad()
                # h = net.init_hidden(self.batch_size).to(device)
                outputs, h = net(train_x_batch.float(), h)

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
                    confusion_matrix_train[tl, pl] = confusion_matrix_train[tl, pl] + 1

            epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Test the model
            acc_test, confusion_matrix_test = self.test_model_conv_gru(net, test_x, test_y, bi_dir, device, confusion_matrix_test)
            test_history.append(acc_test)
            print("\n")


        return train_history, test_history, confusion_matrix_train, confusion_matrix_test

    def test_model_conv_lstm(self, net, test_x, test_y, bi_dir, device, confusion_matrix_test, do_print=True):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            num_lstm_layers = 1
            hidden_dim_of_lstm1 = 256

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
                    confusion_matrix_test[tl, pl] = confusion_matrix_test[tl, pl] + 1

        acc_test = (correct / total) * 100

        if do_print:
            print('Test Accuracy: {} %'.format(acc_test))

        # Confusion matrix
        # print("Test Confusion matrix ")
        # classes = ['W', 'N1', 'N2', 'N3', 'REM']
        # confusion_matrix_accuracy = (confusion_matrix.diag().numpy() / confusion_matrix.sum(1).numpy()) * 100
        # for i, cl in enumerate(classes):
        #     print("F1 ", cl, "{0:.2f}".format(confusion_matrix_accuracy[i]), '%')

        # Test the model
        return acc_test, confusion_matrix_test

    def train_model_conv_lstm(self, net, train_x, train_y, test_x, test_y, bi_dir, device, do_balance=False, do_print=True):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        if do_balance:
            weights = [0.5, 0.5, 1.0, 0.5, 0.5]
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        start = time.time()

        train_history = []
        test_history = []
        num_lstm_layers = 1
        hidden_dim_of_lstm1 = 256

        confusion_matrix_train_list = []
        confusion_matrix_test_list = []

        for epoch in range(1, self.epochs+1):
            net.train()
            if not epoch % 10:
                lr = lr / 1.2

            optimizer = torch.optim.SGD(net.parameters(), lr=lr)

            running_loss = 0
            loss_list = []
            acc_list = []
            correct_all = 0
            total_all = 0
            confusion_matrix_train = torch.zeros(5, 5, dtype=torch.int32)
            confusion_matrix_test = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                # print(train_x_batch.shape)
                # print(train_y_batch.shape)
                # sys.exit()

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                # initial hidden states
                h = torch.zeros((1 + int(bi_dir)) * num_lstm_layers, 1, hidden_dim_of_lstm1)  # if bi_dir=True, first dimension is 2 for bi-directional
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
                    confusion_matrix_train[tl, pl] = confusion_matrix_train[tl, pl] + 1

            epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            if do_print:
                print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Test the model
            acc_test, confusion_matrix_test = self.test_model_conv_lstm(net, test_x, test_y, bi_dir, device, confusion_matrix_test, do_print=do_print)
            test_history.append(acc_test)
            if do_print:
                print("\n")

            confusion_matrix_train_list.append(confusion_matrix_train)
            confusion_matrix_test_list.append(confusion_matrix_test)

        return train_history, test_history, confusion_matrix_train_list, confusion_matrix_test_list

    def test_model_cnn_tsne_draw(self, net, test_x, test_y, device, confusion_matrix_test, do_print=True):

        net.eval()
        conv_output = torch.zeros((test_x.shape[0],  128))
        total_y_batches = torch.zeros((test_y.shape[0]))
        with torch.no_grad():
            correct = 0
            total = 0
            count = 0
            for i in range(0, len(test_x), self.batch_size):
                test_x_batch = test_x[i:i + self.batch_size]
                test_y_batch = test_y[i:i + self.batch_size]

                test_x_batch = torch.from_numpy(test_x_batch).to(device)
                test_y_batch = torch.from_numpy(test_y_batch).view(-1).to(device)

                m_input = test_x_batch
                outputs, lin1 = net(m_input.float())
                conv_output[ i:i+self.batch_size,:] = lin1
                total_y_batches[i:i+self.batch_size] = test_y_batch
                count +=1
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
                    confusion_matrix_test[tl, pl] = confusion_matrix_test[tl, pl] + 1

        acc_test = (correct / total) * 100
        if do_print:
            print('Test Accuracy: {} %'.format(acc_test))

        return acc_test, confusion_matrix_test, conv_output, total_y_batches


    def train_model_cnn_tsne_draw(self, net, train_x, train_y, test_x, test_y, device, do_balance=False, do_print=True):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()

        if do_balance:
            weights = [0.5, 0.5, 1.0, 0.5, 0.5]
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_history = []
        test_history = []
        confusion_matrix_train_list = []
        confusion_matrix_test_list = []
        conv_output_list = []
        total_y_batches_list = []

        for epoch in range(1, self.epochs+1):

            if not epoch % 10:
                lr = lr / 1.0

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            running_loss = 0
            loss_list = []
            acc_list = []
            correct_all = 0
            total_all = 0
            confusion_matrix_train = torch.zeros(5, 5, dtype=torch.int32)
            confusion_matrix_test = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                # train_x_batch = (train_x_batch - mean)/std
                outputs, _ = net(train_x_batch.float())
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
                    confusion_matrix_train[tl, pl] = confusion_matrix_train[tl, pl] + 1

            # epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            if do_print:
                print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Test the model
            acc_test, confusion_matrix_test, conv_output, total_y_batches = self.test_model_cnn_tsne_draw(net, test_x, test_y, device, confusion_matrix_test, do_print=do_print)
            test_history.append(acc_test)
            conv_output_list.append(conv_output)
            total_y_batches_list.append(total_y_batches)
            if do_print:
                print("\n")

            confusion_matrix_train_list.append(confusion_matrix_train)
            confusion_matrix_test_list.append(confusion_matrix_test)

        return train_history, test_history, confusion_matrix_train_list, confusion_matrix_test_list, conv_output_list, total_y_batches_list


    def test_model_cnn(self, net, test_x, test_y, device, confusion_matrix_test, do_print=True):

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0

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
                    confusion_matrix_test[tl, pl] = confusion_matrix_test[tl, pl] + 1

        acc_test = (correct / total) * 100
        if do_print:
            print('Test Accuracy: {} %'.format(acc_test))

        return acc_test, confusion_matrix_test

    def train_model_cnn(self, net, train_x, train_y, test_x, test_y, device, do_balance=False, do_print=True):
        lr = self.learning_rate
        criterion = nn.CrossEntropyLoss()

        if do_balance:
            weights = [0.5, 0.5, 1.0, 0.5, 0.5]
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_history = []
        test_history = []
        confusion_matrix_train_list = []
        confusion_matrix_test_list = []

        for epoch in range(1, self.epochs + 1):

            if not epoch % 10:
                lr = lr / 1.0

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            running_loss = 0
            loss_list = []
            acc_list = []
            correct_all = 0
            total_all = 0
            confusion_matrix_train = torch.zeros(5, 5, dtype=torch.int32)
            confusion_matrix_test = torch.zeros(5, 5, dtype=torch.int32)

            for i in range(0, train_x.shape[0], self.batch_size):
                train_x_batch = train_x[i:i + self.batch_size]
                train_y_batch = train_y[i:i + self.batch_size]

                train_x_batch = torch.from_numpy(train_x_batch).to(device)
                train_y_batch = torch.from_numpy(train_y_batch).view(-1).to(device)

                # train_x_batch = (train_x_batch - mean)/std
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
                    confusion_matrix_train[tl, pl] = confusion_matrix_train[tl, pl] + 1

            # epoch_time = (time.time() - start) / 60
            acc_train = (correct_all / total_all) * 100

            train_history.append(acc_train)
            if do_print:
                print('Epoch [{}/{}]'.format(epoch, self.epochs), ", Accuracy : ", str((correct_all / total_all) * 100))

            # Test the model
            acc_test, confusion_matrix_test = self.test_model_cnn(net, test_x, test_y, device, confusion_matrix_test,
                                                                  do_print=do_print)
            test_history.append(acc_test)
            if do_print:
                print("\n")

            confusion_matrix_train_list.append(confusion_matrix_train)
            confusion_matrix_test_list.append(confusion_matrix_test)

        return train_history, test_history, confusion_matrix_train_list, confusion_matrix_test_list