import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import random
import os
from model import simpleCNN, doubleCNN, fashionCNN

from data_preprocess import preprocess_data
from sklearn.model_selection import KFold

# trainfile_idx = [0, 3, 4, 5]
# testfile_idx = [1, 2]

trainfile_idx = [0, 1, 2, 3]
testfile_idx = [4, 5]
x_train, y_train, test_data, test_labels = preprocess_data('data/eeg_fz_ler', trainfile_idx, testfile_idx)

model = keras.models.load_model('result/output/bestModel/best_doubleCNNmodel')
test_data = test_data.swapaxes(1, 2)
test_labels = keras.utils.to_categorical(test_labels, 5)
predictions=model.predict(test_data)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('predictions:', predictions)
for i in predictions:
    print(i)
print('Test accuracy:', test_acc)