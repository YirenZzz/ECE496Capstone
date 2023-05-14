# ECE496Capstone

## Dataset
The project used the MASS-SS3 dataset. 

## Software
### Data Preprocessing
The raw EEG data is in edf file format and consisted of an Annotated dataset and a PSG dataset. We converted the PSG.edf into multiple 30s segments with a 256Hz sampling as the input x, the data in corresponding Annotation.edf as label y. After encoding and removing any dirty data, the processed data were saved in npz file format. The npz files were then stored in ```Software/data/{cetain_channel}/``` directory.

### Model Building and Training
The model was built and trained using a concatenation of CNNs and BiLSTM. 

1. Pretrain the CNN model by running
```
pretrain.py --data_dir data/{cetain_channel}/ --k_folds {k_folds} --epochs {epochs} --batch_size {batch_size} --model_mode {model_mode}
```
Results from pretraining were saved in Software/output/{best_CNN}/ directory. 

2. Concatenate the outputs with the BiLSTM model for the fine-tuning process by running 
```
fineTune.py --data_dir Software/output/{best_CNN}/ --k_folds {k_folds} --epochs {epochs} --batch_size {batch_size} 
```
Results from fineTune were saved in Software/output/{bestModel}/ directory. 

## Hardware Source Code References

- https://github.com/atomic14/tensorflow-lite-esp32

- https://randomnerdtutorials.com/esp32-vs-code-platformio-spiffs/
