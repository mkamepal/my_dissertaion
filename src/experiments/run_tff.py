import os
import gc
import tensorflow.keras.backend as K
import random
import numpy as np
import tensorflow as tf

from src.config import constants

from src.experiments.run_config import get_config
from src.data_ops.client_partition import prepare_client_dataset
from src.federated.tff_process import build_federated_process
from src.defense.weight_utils import calculate_client_deltas
from src.defense.magnitude import calculate_magnitudes
from src.defense.direction import calculate_head_client_angles
from src.defense.clustering import calculate_hdbscan_results
from src.defense.validation import calculate_validation_results, validation_accuracy_scores
from src.defense.trustscore import calculate_trust_scores
from src.evaluation.model_evaluation import evaluate_global_model

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    K.clear_session()
    gc.collect()
    config = get_config()
    set_seeds(config.seed)

    attack_params = {
        "label_flip": {
            "flip_prob":constants.FLIP_PROB,
            "seed":config.seed
        },

        "backdoor": {
            "target_label":constants.BACKDOOR_TARGET_LABEL,
            "poison_prob":constants.BACKDOOR_POISON_PROB,
            "trigger_value":constants.BACKDOOR_TRIGGER_VALUE,
            "seed":config.seed
        }
    }
    ## Datasets
    (client_datasets,
     client_test_datasets,
     malicious_clients,
     malicious_attack_map) = prepare_client_dataset(
        data_dir=config.data_dir,
        num_client=config.num_clients,
        num_malicious=config.num_malicious,
        attack_list=constants.ATTACK_LIST,
        attack_params=attack_params,
        seed=config.seed,
        max_samples=config.max_samples,
        attack_distribution=constants.ATTACK_DISTRIBUTION,
        test_split=config.test_split,
        batch_size=config.batch_size
    )


    print("\nMalicious clients:")
    print(malicious_clients)
    print("\nAttack map:")
    print(malicious_attack_map)

    ## TFF --> Initialize the server weights for the first time
    input_spec = client_datasets["client_0"].element_spec

    initialize_fn, next_fn = build_federated_process(
            input_spec=input_spec,
            learning_rate=config.learning_rate
        )

    server_weights = initialize_fn()
    client_datasets_list = [client_datasets[f"client_{i}"] for i in range(config.num_clients)]


    ##weights storage variables for the defense methods metrics caluclation

    captured_client_weights = []
    server_weights_before_round = []
    server_weights_after_round = []
    round_losses = []
    round_accuracies = []

    ### Training starts here

    for round_idx in range(config.rounds):

        print("\n" + "=" * 50)
        print(f"ROUND {round_idx + 1}")
        print("=" * 50)

        server_weights_before_round.append(server_weights)
        (server_weights,client_trained_weights,round_metrics) = next_fn(server_weights,client_datasets_list)
        captured_client_weights.append(client_trained_weights)
        server_weights_after_round.append(server_weights)

        loss_value = float(round_metrics["train_loss"])
        accuracy_value = float(round_metrics["train_accuracy"])
        round_losses.append(loss_value)
        round_accuracies.append(accuracy_value)

        print(f"Loss: {loss_value:.4f}")
        print(f"Accuracy: {accuracy_value:.4f}")

    ## Calculating the client deltas
    print("\nCalculating client deltas...")
    client_deltas = calculate_client_deltas(
            captured_client_weights=captured_client_weights,
            server_weights_before_round=server_weights_before_round
        )

    ## Calculating the magnitudes
    print("\nCalculating magnitudes...")
    magnitudes = (calculate_magnitudes(client_deltas))
    for idx in range(len(magnitudes)):
        print(f"round_{idx}: {magnitudes[f'round{idx}weights']}")

    #Clauclating the directions
    print("\nCalculating head-client angles...")
    head_angles = calculate_head_client_angles(client_deltas,head_client="client_0")
    for idx in range(len(head_angles)):
        print(f"round_{idx}: {head_angles[f'round{idx}weights']}")

    ## Performing HDBSCAN
    print("\nRunning HDBSCAN...")
    hdbscan_results = calculate_hdbscan_results(client_weight_deltas=client_deltas,
                                                output_dir=constants.RESULTS_DIR,
                                                total_clients=config.num_clients,
                                                total_malicious=config.num_malicious
                                                )
    for idx in range(len(hdbscan_results)):
        print(print(f"round_{idx}: {hdbscan_results[f'round{idx}weights']}"))

    ## Validation scores

    print("\nEvaluating client models on head-client dataset...")
    # TEMPORARY:
    # Use head test dataset here.
    # Later create separate
    # validation data for client_0.

    head_validation_dataset = client_test_datasets["client_0"]

    validation_results = calculate_validation_results(captured_client_weights = captured_client_weights,
                                                      validation_dataset=head_validation_dataset
                                                      )


    validation_scores = validation_accuracy_scores(validation_results)
    ### Final Global Model
    final_results = evaluate_global_model(server_weights = server_weights,
                                          test_dataset= client_test_datasets["client_0"]
                                          )

    print("\nFinal Global Model")
    print(final_results)
    print("\nExperiment completed.")

    print("\nCalculating trust scores...")
    print("\nCalculating trust scores...")

    print("\nCalculating trust scores...")

    trust_scores = calculate_trust_scores(
        hdbscan_results=hdbscan_results,
        head_angles=head_angles,
        magnitudes=magnitudes,
        validation_results=validation_results,
        head_client="client_0"
    )

if __name__ == "__main__":
    main()