from dataclasses import dataclass
from src.config import constants


@dataclass
class ExperimentConfig:

    data_dir: str
    num_clients: int
    num_malicious: int
    rounds: int
    learning_rate: float
    batch_size: int
    test_split: float
    max_samples: int
    seed: int
    trust_threshold: float
    clustering_weight: float
    direction_weight: float
    magnitude_weight: float
    validation_weight: float


def get_config():
    return ExperimentConfig(data_dir=constants.DATA_DIR,
                            num_clients=constants.NUM_CLIENTS,
                            num_malicious=constants.NUM_MALICIOUS,
                            rounds=constants.FEDERATED_ROUNDS,
                            learning_rate=constants.CLIENT_LEARNING_RATE,
                            batch_size=constants.BATCH_SIZE,
                            test_split=constants.TEST_SPLIT,
                            max_samples=constants.MAX_SAMPLES,
                            seed=constants.SEED,
                            trust_threshold=constants.TRUST_THRESHOLD,
                            clustering_weight=constants.CLUSTERING_WEIGHT,
                            direction_weight=constants.DIRECTION_WEIGHT,
                            magnitude_weight=constants.MAGNITUDE_WEIGHT,
                            validation_weight=constants.VALIDATION_WEIGHT
                            )
