# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import random
import os
from sleep_stage import print_n_samples_each_class

def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate
    
def load_npz_list_files(data_dir, npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        # print("Loading {} ...".format(npz_f))
        # print("npz_f.....",npz_f)
        npz_file = data_dir + '/' + npz_f
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_file)
        
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        data.append(tmp_data)
        labels.append(tmp_labels)

    data = np.vstack(data)
    labels = np.hstack(labels)
    return data, labels


def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:

        idx = np.where(y == c)[0]
        # print('c',c)
        # print('idx',idx)
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        # print('n_repeats',n_repeats)
        # print('x',x)
        # print('len(x)',len(x))
        # print('x[idx]',x[idx])
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def preprocess_data(data_dir, trainfile_idx, testfile_idx):
    # Load all files
    # print('data_dir:',data_dir)
    allfiles = os.listdir(data_dir)
    
    train_files = []
    test_files = []
    
    for idx in trainfile_idx:
        train_files.append(allfiles[idx])
        
    for idx in testfile_idx:
        test_files.append(allfiles[idx])
    # # Ns: the number of subjects in the dataset
    # Ns = int(len(allfiles) / k_folds)
    # print('Ns:',Ns)
    
    # Split train and test files
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    
    # print("test_files:",test_files)
    
    # Load data in npz files
    # data_train: (2796, 1, 7680)
    # label_train: (2796,)
    data_train, label_train = load_npz_list_files(data_dir,train_files)
    # x_val: (2884, 1, 7680)
    # y_val: (2884,)
    x_val, y_val = load_npz_list_files(data_dir, test_files)
    # print('data_train',data_train.shape) # 
    # print('x_val',x_val.shape) # (2884, 1, 7680)
    # print('label_train',label_train.shape) # (2796,)
    # print('y_val',y_val.shape) # (2884,)
    
    # Reshape the data to match the input of the model - conv2d
    data_train = np.squeeze(data_train) # (3868, 7680)
    x_val = np.squeeze(x_val) # (1812, 7680)
    data_train = data_train[:, :, np.newaxis, np.newaxis] # (3868, 7680, 1, 1)
    x_val = x_val[:, :, np.newaxis, np.newaxis] # (1812, 7680, 1, 1)
    
    # Casting
    data_train = data_train.astype(np.float32)
    label_train = label_train.astype(np.int32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.int32)
        
    # print('reshaped data_train',data_train.shape)
    # print('reshaped x_val',x_val.shape)
    
    # print("Training set: {}, {}".format(data_train.shape, label_train.shape))
    # print_n_samples_each_class(label_train)
    # print(" ")
    # print("Validation set: {}, {}".format(x_val.shape, y_val.shape))
    # print_n_samples_each_class(y_val)
    # print(" ")
    
    # Use balanced-class, oversample training set
    x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
    )
    # print("Oversampled training set: {}, {}".format(
    #     x_train.shape, y_train.shape
    # ))
    print_n_samples_each_class(y_train)
    print(" ")
    return x_train, y_train, x_val, y_val
