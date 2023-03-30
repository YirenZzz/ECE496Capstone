import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="example_model.tflite")
interpreter.allocate_tensors()

# Print input tensor shape
input_details = interpreter.get_input_details()[0]
print("Input shape:", input_details['shape'])

# Print output tensor shape
output_details = interpreter.get_output_details()[0]
print("Output shape:", output_details['shape'])