import tensorflow as tf
import keras.layers as kl
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Reshape
#Example
# class CNNModel(keras.Model):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = Conv2D(32, (3,3), activation='relu')
#         self.pool1 = MaxPooling2D(pool_size=(2, 2))
#         self.flatten = Flatten()
#         self.dense1 = Dense(128, activation='relu')
#         self.dense2 = Dense(10, activation='softmax')

#     def call(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         return self.dense2(x)


class simpleCNN(keras.Model):

    def __init__(self):
        super(simpleCNN, self).__init__()
        
        # Convolution
        # network = conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)
        self.conv1 = kl.Conv1D(filters=8, kernel_size=25, strides=3,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.ReLU()
        # Max pooling
        # network = max_pool_1d(input_var=network, pool_size=8, stride=8)
        self.pool1 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(80, 1, 8))
        # self.pool1 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same')
        # Dropout
        self.drop1 = kl.Dropout(rate=0.5)
        
        # Convolution
        self.conv2 = kl.Conv1D(filters=4, kernel_size=2, strides=1,
                        padding='same', activation='relu')
        self.bn2 = kl.BatchNormalization()
        self.relu2 = kl.ReLU()
        
        self.conv3 = kl.Conv1D(filters=4, kernel_size=2, strides=1,
                        padding='same', activation='relu')
        self.bn3 = kl.BatchNormalization()
        self.relu3 = kl.ReLU()
        
        self.conv4 = kl.Conv1D(filters=4, kernel_size=2, strides=1,
                        padding='same', activation='relu')
        self.bn4 = kl.BatchNormalization()
        self.relu4 = kl.ReLU()
        # Max pooling
        # self.pool2 = kl.MaxPooling1D(pool_size=4, strides=4)
        # self.pool2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same')
        self.pool2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(16, 1, 8))
        
        # Flatten
        self.flat = kl.Flatten()

        # Softmax layer
        self.dense = kl.Dense(units=5, activation='softmax')

    def call(self, x, training=True):

        print('xxx',x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        print('x c1',x.shape)
        # x = self.pool1(x)
        print('x pool1',x.shape)
    
        if training:
            x = self.drop1(x, training=True)
            
        # x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        print('x c2',x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        print('x c3',x.shape)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        print('x c4',x.shape)
        # x = self.pool2(x) # (batch, 1, 40, 128)
        print('x p2',x.shape)
        x = self.flat(x) #(5, 5120)
        if training:
            x = self.drop1(x, training=True)

        print('x fc',x.shape) 
        x = self.dense(x) #(5, 5)
        print('x dense',x.shape)
        return x
    
    def compile(self, optimizer, loss, metrics):
        super(simpleCNN, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)

class doubleCNN(keras.Model):

    def __init__(self):
        super(doubleCNN, self).__init__()
        
        # Convolution
        # network = conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)
        self.conv1 = kl.Conv1D(filters=64, kernel_size=50, strides=6,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.ReLU()
        # Max pooling
        # network = max_pool_1d(input_var=network, pool_size=8, stride=8)
        self.pool1 = kl.MaxPooling2D(pool_size=(1,8), strides=(1,8),padding='same',input_shape=(1280, 1, 64))
        
        # Dropout
        self.drop1 = kl.Dropout(rate=0.5)
        
        
        # Convolution
        self.conv2 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn2 = kl.BatchNormalization()
        self.relu2 = kl.ReLU()
        
        self.conv3 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn3 = kl.BatchNormalization()
        self.relu3 = kl.ReLU()
        
        self.conv4 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn4 = kl.BatchNormalization()
        self.relu4 = kl.ReLU()
        # Max pooling
        # self.pool2 = kl.MaxPooling1D(pool_size=4, strides=4)
        self.pool2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(160, 1, 128))
        
        # Flatten
        self.flat = kl.Flatten()

        ######### CNNs with large filter size at the first layer #########
        
        self.conv1_2 = kl.Conv1D(filters=64, kernel_size=400, strides=50,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.bn1_2 = kl.BatchNormalization()
        self.relu1_2 = kl.ReLU()
        # Max pooling
        # network = max_pool_1d(input_var=network, pool_size=8, stride=8)
        self.pool1_2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(1280, 1, 64))
        
        # Dropout
        self.drop1_2 = kl.Dropout(rate=0.5)
        
        # Convolution
        self.conv2_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn2_2 = kl.BatchNormalization()
        self.relu2_2 = kl.ReLU()
        
        self.conv3_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn3_2 = kl.BatchNormalization()
        self.relu3_2 = kl.ReLU()
        
        self.conv4_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn4_2 = kl.BatchNormalization()
        self.relu4_2 = kl.ReLU()
        # Max pooling
        # self.pool2 = kl.MaxPooling1D(pool_size=4, strides=4)
        self.pool2_2 = kl.MaxPooling2D(pool_size=(1,2), strides=(1,2),padding='same',input_shape=(160, 1, 128))
        
        # Flatten
        self.flat_2 = kl.Flatten()
        self.denseadd = kl.Dense(units=2560, activation='relu')
        # Softmax layer
        self.dense = kl.Dense(units=5, activation='softmax')

    def call(self, input, training=True):
        x = input
        ######### CNNs with small filter size at the first layer #########
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        print('x c1',x.shape)
        x = self.pool1(x)
        print('x pool1',x.shape)
    
        if training:
            x = self.drop1(x, training=True)
            
        # x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        print('x c2',x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        print('x c3',x.shape)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        print('x c4',x.shape)
        x = self.pool2(x) # (batch, 1, 40, 128)
        print('x p2',x.shape)
        x = self.flat(x) #(5, 5120)
        
        # ######### CNNs with large filter size at the first layer #########
        x2 = input
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.relu1_2(x2)
        print('x c1',x.shape)
        x2 = self.pool1_2(x2)
        print('x pool1',x2.shape)
    
        if training:
            x2 = self.drop1_2(x2, training=True)
                        
        # x = self.drop1(x)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.relu2_2(x2)
        print('x c2',x.shape)
        x2 = self.conv3_2(x2)
        x2 = self.bn3_2(x2)
        x2 = self.relu3_2(x2)
        print('x c3',x.shape)
        x2 = self.conv4_2(x2)
        x2 = self.bn4_2(x2)
        x2 = self.relu4_2(x2)
        print('x c4',x2.shape)
        x2 = self.pool2_2(x2) # (batch, 1, 40, 128)
        print('x p2',x2.shape)
        x2 = self.flat(x2) #(5, 5120)
        print('x fc',x2.shape) 
        
        merged = Concatenate(axis=1)([x, x2])
        
        # cat
        if training:
            merged = self.drop1(merged, training=True)

        # merged = self.denseadd(merged) #(5, 5)
        merged = self.dense(merged)
        print('merged',merged.shape)
        return merged
    
    def compile(self, optimizer, loss, metrics):
        super(doubleCNN, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)

class doubleCNNBiLSTM(keras.Model):

    def __init__(self, output_dim, lstm_units):
        super(doubleCNNBiLSTM, self).__init__()
        
        # Convolution
        # network = conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)
        self.conv1 = kl.Conv1D(filters=64, kernel_size=50, strides=6,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.ReLU()
        # Max pooling
        # network = max_pool_1d(input_var=network, pool_size=8, stride=8)
        self.pool1 = kl.MaxPooling2D(pool_size=(1,8), strides=(1,8),padding='same',input_shape=(1280, 1, 64))
        
        # Dropout
        self.drop1 = kl.Dropout(rate=0.5)
        
        
        # Convolution
        self.conv2 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn2 = kl.BatchNormalization()
        self.relu2 = kl.ReLU()
        
        self.conv3 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn3 = kl.BatchNormalization()
        self.relu3 = kl.ReLU()
        
        self.conv4 = kl.Conv1D(filters=128, kernel_size=8, strides=1,
                        padding='same', activation='relu')
        self.bn4 = kl.BatchNormalization()
        self.relu4 = kl.ReLU()
        # Max pooling
        # self.pool2 = kl.MaxPooling1D(pool_size=4, strides=4)
        self.pool2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(160, 1, 128))
        
        # Flatten
        self.flat = kl.Flatten()

        ######### CNNs with large filter size at the first layer #########
        
        self.conv1_2 = kl.Conv1D(filters=64, kernel_size=400, strides=50,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.bn1_2 = kl.BatchNormalization()
        self.relu1_2 = kl.ReLU()
        # Max pooling
        # network = max_pool_1d(input_var=network, pool_size=8, stride=8)
        self.pool1_2 = kl.MaxPooling2D(pool_size=(1,4), strides=(1,4),padding='same',input_shape=(1280, 1, 64))
        
        # Dropout
        self.drop1_2 = kl.Dropout(rate=0.5)
        
        # Convolution
        self.conv2_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn2_2 = kl.BatchNormalization()
        self.relu2_2 = kl.ReLU()
        
        self.conv3_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn3_2 = kl.BatchNormalization()
        self.relu3_2 = kl.ReLU()
        
        self.conv4_2 = kl.Conv1D(filters=128, kernel_size=6, strides=1,
                        padding='same', activation='relu')
        self.bn4_2 = kl.BatchNormalization()
        self.relu4_2 = kl.ReLU()
        # Max pooling
        # self.pool2 = kl.MaxPooling1D(pool_size=4, strides=4)
        self.pool2_2 = kl.MaxPooling2D(pool_size=(1,2), strides=(1,2),padding='same',input_shape=(160, 1, 128))
        
        # Flatten
        self.flat_2 = kl.Flatten()
        
   
        # Bidirectional LSTM layer 1
        self.bi_lstm1 = kl.Bidirectional(kl.LSTM(units=lstm_units, return_sequences=True))
        
        # Bidirectional LSTM layer 2
        self.bi_lstm2 = kl.Bidirectional(kl.LSTM(units=lstm_units, return_sequences=False))
        # Softmax layer
        self.out_layer = kl.Dense(units=output_dim, activation='softmax')

    def call(self, input, training=True):
        x = input
        ######### CNNs with small filter size at the first layer #########
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        print('x c1',x.shape)
        x = self.pool1(x)
        print('x pool1',x.shape)
    
        if training:
            x = self.drop1(x, training=True)
            
        # x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        print('x c2',x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        print('x c3',x.shape)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        print('x c4',x.shape)
        x = self.pool2(x) # (batch, 1, 40, 128)
        print('x p2',x.shape)
        x = self.flat(x) #(5, 5120)
        
        # ######### CNNs with large filter size at the first layer #########
        x2 = input
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.relu1_2(x2)
        print('x c1',x.shape)
        x2 = self.pool1_2(x2)
        print('x pool1',x2.shape)
    
        if training:
            x2 = self.drop1_2(x2, training=True)
                        
        # x = self.drop1(x)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.relu2_2(x2)
        print('x c2',x.shape)
        x2 = self.conv3_2(x2)
        x2 = self.bn3_2(x2)
        x2 = self.relu3_2(x2)
        print('x c3',x.shape)
        x2 = self.conv4_2(x2)
        x2 = self.bn4_2(x2)
        x2 = self.relu4_2(x2)
        print('x c4',x2.shape)
        x2 = self.pool2_2(x2) # (batch, 1, 40, 128)
        print('x p2',x2.shape)
        x2 = self.flat(x2) #(5, 5120)
        print('x fc',x2.shape) 
        
        x = Concatenate(axis=1)([x, x2])
        
        print('x cocat',x.shape) 
        x = self.drop1(x)
        x = Reshape(target_shape=(60, 128))(x)
        
        print('x reshape',x.shape) 
        
        # BiLSTM layers
        x = self.bi_lstm1(x)
        x = self.drop1(x)
        x = self.bi_lstm2(x)
        
        # cat
        if training:
            x = self.drop1(x, training=True)

        # Output layer
        output = self.out_layer(x)
        print('output',output.shape)
        return output
    
    def compile(self, optimizer, loss, metrics):
        super(doubleCNNBiLSTM, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)



class fashionCNN(keras.Model):

    def __init__(self):
        super(fashionCNN, self).__init__()
        
        # self.conv1 = tf.keras.layers.Conv2D(6, 3, activation='relu', input_shape=(1280, 1, 64))
        # self.pool1 = tf.keras.layers.MaxPooling2D()
        # self.conv2 = tf.keras.layers.Conv2D(6, 3, activation='relu')
        # self.flatten = tf.keras.layers.Flatten()
        # self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
        self.conv1 = kl.Conv1D(filters=32, kernel_size=20, strides=6,
                        padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        
        self.pool1 = kl.MaxPooling2D(pool_size=(1,8), strides=(1,8),padding='same',input_shape=(640, 1, 64))
        
        self.conv2 = kl.Conv1D(filters=64, kernel_size=4, strides=1,
                        padding='same', activation='relu')
        
        self.flat = kl.Flatten()
        
        self.dense = kl.Dense(units=5, activation='softmax')

    def call(self, x,  training=True):
        print('x input',x.shape)
        x = self.conv1(x)
        print('x c1',x.shape)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
    
    
    def compile(self, optimizer, loss, metrics):
        super(fashionCNN, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
