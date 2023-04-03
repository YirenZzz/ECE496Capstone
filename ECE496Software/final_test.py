from tensorflow.keras import Input, Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Softmax, LSTM

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional


pre_model = tf.keras.models.load_model('result/output/bestModel/best_doubleCNNmodel')

# Create dummy data
X = np.random.rand(250, 25, 7680,1)
y = np.random.randint(5, size=(250, 5))

# Define the fine-tuning model architecture
input_shape = (None, 3000, 1) # assuming variable-length sequences
input_seq = Input(shape=input_shape)
cnn1 = Conv1D(filters=32, kernel_size=10, activation='relu')(input_seq)
pool1 = MaxPooling1D(pool_size=3)(cnn1)
cnn2 = Conv1D(filters=64, kernel_size=10, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=3)(cnn2)
lstm1 = LSTM(units=128, return_sequences=True)(pool2)
lstm2 = LSTM(units=128)(lstm1)
dense = Dense(units=5, activation='softmax')(lstm2)
model = Model(inputs=input_seq, outputs=dense)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, batch_size=10, epochs=1)
# Print the model summary
model.summary()
