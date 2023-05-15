# ECE496Capstone

## Dataset
This project use the MASS-SS3 dataset. 

## Software
### Data Preprocessing
The raw EEG data is in edf file format and consisted of an Annotated dataset and a PSG dataset. We converted the PSG.edf into multiple 30s segments with a 256Hz sampling as the input x, the data in the corresponding Annotation.edf as label y. After encoding and removing any dirty data(including data with label of ?), the processed data were saved in npz file format and were then stored in ```Software/data/{cetain_channel}/``` directory.

### Model Building and Training
The model was built and trained using a concatenation of CNNs and BiLSTM. 

1. Pretrain the CNN model by running
```
pretrain.py --data_dir data/{cetain_channel}/ --k_folds {k_folds} --epochs {epochs} --batch_size {batch_size} --model_mode {model_mode}
```
Results from pretraining were saved in ```Software/output/{best_CNN}/``` directory. 

2. Concatenate the outputs with the BiLSTM model for the fine-tuning process by running 
```
fineTune.py --data_dir Software/output/{best_CNN}/ --k_folds {k_folds} --epochs {epochs} --batch_size {batch_size} 
```
Results from fineTune were saved in ```Software/output/{bestModel}/``` directory, which will be converted to Tensorflow Lite model for hardware deployment.

## Hardware
### Model Conversion
1.	Tensorflow Lite’s converter is used to convert the Tensorflow model into a Tensorflow Lite model.
2.	The Tensorflow Lite model is converted into a C source file to be compatible for deployment. This is done with the unix xxd command, which makes a hex dump of the Tensorflow Lite model file.

Sample code can be found in ```Hardware_ESP32/Convert_New_Model.ipynb```

### MCU Deployment
1.	Firmware code is written to control the operations of the MCU utilizing methods from the Tensorflow Lite library. 
2.	The model is deployed to the ESP32-S2-DEVKITC-1 MCU which is connected to the computer through a USB-to-UART cable. PlatformIO IDE is used to facilitate the deployment process.

Project source code can be found in ```Hardware_ESP32/```

Code References
- https://github.com/atomic14/tensorflow-lite-esp32

- https://randomnerdtutorials.com/esp32-vs-code-platformio-spiffs/
