rm -rf example_model
rm -rf example_model.tflite
rm -rf output_model.h

python example_model.py
python convert_to_tflite.py
xxd -i example_model.tflite > output_model.h
