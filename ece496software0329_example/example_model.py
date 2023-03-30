import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Define the input shape
input_shape = (1, 7680, 1)

x_train = np.random.randn(100, 1, 7680, 1) #(100, 1, 7680, 1)
y_train = np.random.randint(5, size=(100,)) #(100, 5)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 5)

model = tf.keras.Sequential([
    layers.Conv1D(1, 1, activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((1, 1)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(4, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

# Print the layer shapes
for layer in model.layers:
    print(layer.output_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)
model.save("example_model")