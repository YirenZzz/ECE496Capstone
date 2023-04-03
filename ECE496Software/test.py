# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# class CNNModel(keras.Model):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = Conv2D(32, (3,3), activation='relu')
#         self.pool1 = MaxPooling2D(pool_size=(2, 2))
#         self.flatten = Flatten()
#         self.dense1 = Dense(128, activation='relu')
#         self.dense2 = Dense(10, activation='softmax')

#     def call(self, x):
#         print('xshape',x.shape)
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         print('xshape d2',x.shape)
#         return x

# def train_model(model, x_train, y_train, x_test, y_test):
#     # Compile the model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     print('x_train',x_train.shape)
#     print('y_train',y_train.shape)
#     # Train the model
#     model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))
#     # model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

#     # Evaluate the model on the test data
#     test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
#     print('Test accuracy:', test_accuracy)

# # Load the data
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# # Preprocess the data
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# # Create an instance of the model
# model = CNNModel()


# # Train the model
# train_model(model, x_train, y_train, x_test, y_test)

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="example_model.tflite")
interpreter.allocate_tensors()

# Print input tensor shape
input_details = interpreter.get_input_details()[0]
print("Input shape:", input_details['shape'])

# Print output tensor shape
output_details = interpreter.get_output_details()[0]
print("Output shape:", output_details['shape'])