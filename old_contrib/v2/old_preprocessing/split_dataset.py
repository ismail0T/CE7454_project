"""
This file is to get 2013 data from all data
"""

import os, glob
from shutil import copyfile


dataset_folder = r"E:\sleep-edf-database-expanded-1.0.0"
data_2013_folder = r"E:\data_2013"


data_2013_file = open(os.path.join(dataset_folder, "RECORDS-v1"),'r')
data_2013 = data_2013_file.read()

files = []

# walk through all the files
for r, d, f in os.walk(dataset_folder):
    for file in f:
        if '.edf' in file:
            files.append(os.path.join(r, file))


# copying files into new path
for i in data_2013.split("\n"):
    psg_file = i
    hyp_file = i[:7]
    for j in files:
        if i in j:
            des = os.path.join(data_2013_folder, i)
            copyfile(j, des)
        if (hyp_file in j) and ("Hyp" in j):
            #print(j.split("\"")[-1])
            des = os.path.join(data_2013_folder, j.split("\\")[-1])
            #print(des)
            copyfile(j, des)
    
