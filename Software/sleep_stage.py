import numpy as np

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

NUM_CLASSES = 6  # exclude UNKNOWN

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "?"
}

EPOCH_SEC_LEN = 30  # seconds
SAMPLING_RATE = 256

def print_n_samples_each_class(labels):
    
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        # print("{}: {}".format(class_dict[c], n_samples))
