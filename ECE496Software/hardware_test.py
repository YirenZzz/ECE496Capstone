
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
# from sklearn.model_selection import KFold
import argparse
import random
import os
import tensorflow as tf
from tensorflow import keras
# from model import simpleCNN, doubleCNN, fashionCNN

from data_preprocess import preprocess_data
from sklearn.model_selection import KFold


# def representative_data_gen():

#   trainfile_idx = [0,3,4,5]
#   testfile_idx = [1,2]
#   data_dir = "./data/eeg_fz_ler"

#   x_train, y_train, x_val, y_val = preprocess_data(data_dir, trainfile_idx, testfile_idx)
#   test_batches_ori = (x_val, y_val)
# #   input_value = x_val[:1, :]
# #   print(input_value.shape)
#   x_val = x_val.swapaxes(1, 2)
#   for i in range(101):
#   #   yield [input_value]
#     print(x_val[i:i+1, :].shape)
#     yield [x_val[i:i+1, :]]

# # representative_data_gen()
# export_dir = './result/output/bestModel/best_fashionCNNmodel'
# tflite_model_file = 'OPTbest_fashionCNNmodel.tflite'

# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
# tflite_model = converter.convert()


model = tf.keras.models.load_model('./result/output/bestModel/best_fashionCNNmodel')
# input_tensor = model.inputs
# shape_tensor = tf.shape( input_tensor )
print(model.summary())

print("model.inputs",model.input)

# # Attempt to fix reshape error
# batch_size = 1
# input_shape = model.inputs[0].shape.as_list()
# input_shape[0] = batch_size

# func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))
# model_converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
# model_lite = model_converter.convert()

# f = open("./FixReshapeError.tflite", "wb")
# f.write(model_lite)
# f.close()
# print('shape', shape_tensor)
