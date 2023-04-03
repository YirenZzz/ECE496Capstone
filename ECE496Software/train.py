import tensorflow as tf
from tensorflow import keras


def train_model(model, x_train, y_train, batch_size, epochs):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

def test_model(model, x_test, y_test, batch_size):
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', test_accuracy)