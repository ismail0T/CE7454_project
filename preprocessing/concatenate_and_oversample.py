import os, time
from datetime import datetime
import numpy as np
import argparse

from dataloader import SeqDataLoader

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="E:/data_2013",
                    help="File path to the generated numpy files.")
parser.add_argument("--sampling", type=str, default="smote",
                    help="Required sampling technique: either smote or rus")


args = parser.parse_args()

num_folds = 20  #CHANGE THIS


##data_dir = r"E:\data_2013_npz\eeg_fpz_cz"


classes = ["W", "N1", "N2", "N3", "REM"]
n_classes = len(classes)


max_time_step=1#10,  # 5 3 second best 10# 40 # 100

channel_ename = ""  # CHANGE THIS
path = os.path.split(args.data_dir)
output_dir = "traindata_" + args.sampling
traindata_dir = os.path.join(args.data_dir, output_dir)
if not os.path.exists(traindata_dir):
    os.makedirs(traindata_dir)

print(str(datetime.now()))


for fold_idx in range(num_folds):
    start_time_fold_i = time.time()
    data_loader = SeqDataLoader(args.data_dir, num_folds, fold_idx, classes=classes)
    X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=max_time_step)

    # preprocessing
    char2numY = dict(zip(classes, range(len(classes))))

##    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
##    char2numY['<SOD>'] = len(char2numY)
##    char2numY['<EOD>'] = len(char2numY)
##    num2charY = dict(zip(char2numY.values(), char2numY.keys()))


    # over-sampling: SMOTE:
    X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
    y_train= y_train.flatten()

    nums = []
    for cl in classes:
        nums.append(len(np.where(y_train == char2numY[cl])[0]))

    if (os.path.exists(traindata_dir) == False):
        os.mkdir(traindata_dir)

    fname = os.path.join(traindata_dir,'trainData_' + channel_ename + '_' + args.sampling + '_all_10s_f'+str(fold_idx)+'.npz')
    fname_test = os.path.join(traindata_dir, 'trainData_' + channel_ename + '_' + args.sampling + '_all_10s_f' + str(fold_idx) + '_TEST.npz')

    if (os.path.isfile(fname)):
        X_train, y_train,_ = data_loader.load_npz_file(fname)

    else:
        
        n_osamples = nums[2] - 7000
        ratio = {0: n_osamples if nums[0] < n_osamples else nums[0],
                 1: n_osamples if nums[1] < n_osamples else nums[1],
                 2: nums[2],
                 3: n_osamples if nums[3] < n_osamples else nums[3],
                 4: n_osamples if nums[4] < n_osamples else nums[4]}


        if args.sampling == "smote":
            sm = SMOTE(random_state=12,ratio=ratio)
            X_train, y_train = sm.fit_sample(X_train, y_train)
        elif args.sampling == "rus":
            rus = RandomUnderSampler(random_state=12)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        else:
            pass

        data_loader.save_to_npz_file(X_train, y_train,data_loader.sampling_rate,fname)
        data_loader.save_to_npz_file(X_test, y_test, data_loader.sampling_rate, fname_test)
