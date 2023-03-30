import tensorflow as tf

# Load the Keras model
# model = tf.keras.models.load_model("result/output/bestModel/best_simpleCNNmodel")
model = tf.keras.models.load_model("example_model")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
with open("example_model.tflite", 'wb') as f:
    f.write(tflite_model)