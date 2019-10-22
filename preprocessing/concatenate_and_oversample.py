import os, time
from datetime import datetime
import numpy as np

from dataloader import SeqDataLoader

from imblearn.over_sampling import SMOTE


num_folds = 8  #CHANGE THIS
data_dir = "E:\data_2013_npz\eeg_fpz_cz"

if '13' in data_dir:
    data_version = 2013
else:
    n_oversampling = 30000
    data_version = 2018

output_dir = "E:\data_2013_output"
classes = ["W", "N1", "N2", "N3", "REM"]
n_classes = len(classes)

hparams = dict(
        epochs=10,
        batch_size=20,  # 10
        num_units=128,
        embed_size=10,
        input_depth=300,#3000,
        n_channels=100,
        bidirectional=False,
        use_attention=True,
        lstm_layers=2,
        attention_size=64,
        beam_width=4,
        use_beamsearch_decode=False,
        max_time_step=1,#10,  # 5 3 second best 10# 40 # 100
        output_max_length=10 + 2,  # max_time_step +1
        akara2017=True,
        test_step=5  # each 10 epochs
    )


channel_ename = ""  # CHANGE THIS
path = os.path.split(data_dir)
traindata_dir = os.path.join(os.path.abspath(os.path.join(data_dir, os.pardir)),'traindata4/')
print(str(datetime.now()))


for fold_idx in range(num_folds):
    start_time_fold_i = time.time()
    data_loader = SeqDataLoader(data_dir, num_folds, fold_idx, classes=classes)
    X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=hparams["max_time_step"])

    # preprocessing
    char2numY = dict(zip(classes, range(len(classes))))
    pre_f1_macro = 0

    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
    char2numY['<SOD>'] = len(char2numY)
    char2numY['<EOD>'] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))


    # over-sampling: SMOTE:
    X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
    y_train= y_train.flatten()

    if data_version == 2018:
        # extract just undersamples For 2018
        under_sample_len = 35000#30000
        Ws = np.where(y_train == char2numY['W'])[0]
        len_W = len(np.where(y_train == char2numY['W'])[0])
        permute = np.random.permutation(len_W)
        len_r = len_W - under_sample_len if (len_W - under_sample_len) > 0 else 0
        permute = permute[:len_r]
        y_train = np.delete(y_train,Ws[permute],axis =0)
        X_train = np.delete(X_train,Ws[permute],axis =0)

        under_sample_len = 35000 #40000
        N2s = np.where(y_train == char2numY['N2'])[0]
        len_N2 = len(np.where(y_train == char2numY['N2'])[0])
        permute = np.random.permutation(len_N2)
        len_r = len_N2 - under_sample_len if (len_N2 - under_sample_len) > 0 else 0
        permute = permute[:len_r]
        y_train = np.delete(y_train, N2s[permute],axis =0)
        X_train = np.delete(X_train, N2s[permute],axis =0)

    nums = []
    for cl in classes:
        nums.append(len(np.where(y_train == char2numY[cl])[0]))

    if (os.path.exists(traindata_dir) == False):
        os.mkdir(traindata_dir)
    fname = os.path.join(traindata_dir,'trainData_'+channel_ename+'_SMOTE_all_10s_f'+str(fold_idx)+'.npz')
    fname_test = os.path.join(traindata_dir, 'trainData_' + channel_ename + '_SMOTE_all_10s_f' + str(fold_idx) + '_TEST.npz')
    if (os.path.isfile(fname)):
        X_train, y_train,_ = data_loader.load_npz_file(fname)

    else:
        if data_version == 2013:
            n_osamples = nums[2] - 7000
            ratio = {0: n_osamples if nums[0] < n_osamples else nums[0], 1: n_osamples if nums[1] < n_osamples else nums[1],
                     2: nums[2], 3: n_osamples if nums[3] < n_osamples else nums[3], 4: n_osamples if nums[4] < n_osamples else nums[4]}


        if data_version==2018:
            ratio = {0: n_oversampling if nums[0] < n_oversampling else nums[0], 1: n_oversampling if nums[1] < n_oversampling else nums[1], 2: nums[2],
                 3: n_oversampling if nums[3] < n_oversampling else nums[3], 4: n_oversampling if nums[4] < n_oversampling else nums[4]}

        # ratio = {0: 40000 if nums[0] < 40000 else nums[0], 1: 27000 if nums[1] < 27000 else nums[1], 2: nums[2],
        #          3: 30000 if nums[3] < 30000 else nums[3], 4: 27000 if nums[4] < 27000 else nums[4]}
        sm = SMOTE(random_state=12,ratio=ratio)
        # sm = SMOTE(random_state=12, ratio=ratio)
        # sm = RandomUnderSampler(random_state=12,ratio=ratio)
        X_train, y_train = sm.fit_sample(X_train, y_train)
        data_loader.save_to_npz_file(X_train, y_train,data_loader.sampling_rate,fname)
        data_loader.save_to_npz_file(X_test, y_test, data_loader.sampling_rate, fname_test)

#     X_train = X_train[:(X_train.shape[0] // hparams["max_time_step"]) * hparams["max_time_step"], :]
#     y_train = y_train[:(X_train.shape[0] // hparams["max_time_step"]) * hparams["max_time_step"]]

#     X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
#     y_train = np.reshape(y_train,[-1,y_test.shape[1],])

#     # shuffle training data_2013
#     permute = np.random.permutation(len(y_train))
#     X_train = np.asarray(X_train)
#     X_train = X_train[permute]
#     y_train = y_train[permute]


#     # add '<SOD>' to the beginning of each label sequence, and '<EOD>' to the end of each label sequence (both for training and test sets)
#     y_train= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_train]
#     y_train = np.array(y_train)


#     y_test= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_test]
#     y_test = np.array(y_test)

#     print ('The training set after oversampling: ', classes)
#     for cl in classes:
#         print (cl, len(np.where(y_train==char2numY[cl])[0]))

#     #% run CNN and lstm
#         # evalaute
        
