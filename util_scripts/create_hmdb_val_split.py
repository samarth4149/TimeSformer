import os
import sys
sys.path.append(os.path.abspath('.'))
from pathlib import Path
import csv
import numpy as np

if __name__ == '__main__':
    df_path_old = Path('data_files/hmdb51')
    df_path_new = Path('data_files/hmdb51_val')
    df_path_new.mkdir(exist_ok=True)
    separator = ' '
    val_prop = 0.2
    RNG = np.random.RandomState(44)
    
    tr_files = []
    tr_labels = []
    with open(df_path_old / 'train.csv', 'r') as f:
        reader = csv.reader(f, delimiter=separator)
        for row in reader:
            tr_files.append(row[0])
            tr_labels.append(row[1])
    tr_labels = np.array(tr_labels)
    tr_files = np.array(tr_files)

    for cl in np.unique(tr_labels):
        cl_files = tr_files[tr_labels==cl]
        cl_files = RNG.permutation(cl_files)
        curr_val_files = cl_files[:int(val_prop*len(cl_files))]
        curr_tr_files = cl_files[int(val_prop*len(cl_files)):]
        with open(df_path_new / 'train.csv', 'a+') as f:
            writer = csv.writer(f, delimiter=separator)
            for file in curr_tr_files:
                writer.writerow([file, cl])
        with open(df_path_new / 'val.csv', 'a+') as f:
            writer = csv.writer(f, delimiter=separator)
            for file in curr_val_files:
                writer.writerow([file, cl])