# config/constants.py


# ============================================================
# FEDERATED LEARNING
# ============================================================

NUM_CLIENTS = 15

HEAD_CLIENT = "client_0"

FEDERATED_ROUNDS = 30

STORE_RESULTS = True


## Malicious client configurations

NUM_MALICIOUS = 9

MAX_FAKE_RATIO = 0.7

## Attack Configurations
LABEL_FLIP = True
FLIP_PROB = 0.4

BACKDOOR = True
BACKDOOR_POISON_PROB = 0.4
BACKDOOR_TARGET_LABEL = 0
BACKDOOR_TRIGGER_VALUE = 1.0

ATTACK_LIST = [
    "label_flip",
    "backdoor"
]

ATTACK_DISTRIBUTION = {
    "label_flip": 0.60,
    "backdoor": 0.40
}


# Reproducibility
SEED = 42


## Image Dataset Configurations

IMAGE_SIZE = (224, 224)
TARGET_SIZE = (64, 64)
TEST_SPLIT = 0.12
BATCH_SIZE = 8
MAX_SAMPLES = 10000

## Machine Learning Parameters
CLIENT_LEARNING_RATE = 0.01


## Trust Score weights and threshold

CLUSTERING_WEIGHT = 0.25
DIRECTION_WEIGHT = 0.25
MAGNITUDE_WEIGHT = 0.25
VALIDATION_WEIGHT = 0.25

TRUST_THRESHOLD = 0.50

## Data Dir path
DATA_DIR = '/home/mahesh/Desktop/Dissertation/tf_fd_cancer_fakeclients_detection/data/Kidney Cancer'

## results dir
RESULTS_DIR = "/home/mahesh/Desktop/Dissertation/tf_fd_cancer_fakeclients_detection/src/results"


TRUST_WEIGHT_CLUSTER = 0.25
TRUST_WEIGHT_DIRECTION = 0.25
TRUST_WEIGHT_MAGNITUDE = 0.25
TRUST_WEIGHT_VALIDATION = 0.25

# Angle in degrees
DIRECTION_THRESHOLD = 30.0

# Maximum allowed relative magnitude difference
# 0.20 = 20%
MAGNITUDE_THRESHOLD = 0.20

# Maximum allowed accuracy drop from head client
# 0.10 = 10 percentage points
VALIDATION_ACCURACY_TOLERANCE = 0.1

USE_TRUST_FILTERING = True
