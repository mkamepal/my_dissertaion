import os
import tensorflow as tf

from openpyxl import Workbook
from src.models.cnn import create_keras_model
from src.config import constants


# ============================================================
# SAVE VALIDATION RESULTS TO EXCEL
# ============================================================

def save_validation_results_to_excel(
    validation_results
):
    """
    Save validation loss and accuracy for every client
    in every federated round.

    Columns:
        Round
        Client ID
        Validation Loss
        Validation Accuracy
    """

    output_dir = constants.RESULTS_DIR

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    file_name = (
        f"validation"
        f"_clients_{constants.NUM_CLIENTS}"
        f"_malicious_{constants.NUM_MALICIOUS}"
        f"_rounds_{constants.FEDERATED_ROUNDS}.xlsx"
    )

    file_path = os.path.join(
        output_dir,
        file_name
    )

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = (
        "Validation Results"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    worksheet.append([
        "Round",
        "Client ID",
        "Validation Loss",
        "Validation Accuracy"
    ])

    # --------------------------------------------------------
    # Store all validation results
    # --------------------------------------------------------

    for (
        round_idx,
        (
            round_key,
            client_results
        )
    ) in enumerate(
        validation_results.items(),
        start=1
    ):

        client_ids = sorted(
            client_results.keys(),
            key=lambda value: int(
                value.split("_")[1]
            )
        )

        for client_id in client_ids:

            result = client_results[
                client_id
            ]

            worksheet.append([
                round_idx,
                client_id,
                float(
                    result["loss"]
                ),
                float(
                    result["accuracy"]
                )
            ])

    # --------------------------------------------------------
    # Column widths
    # --------------------------------------------------------

    worksheet.column_dimensions[
        "A"
    ].width = 12

    worksheet.column_dimensions[
        "B"
    ].width = 18

    worksheet.column_dimensions[
        "C"
    ].width = 22

    worksheet.column_dimensions[
        "D"
    ].width = 24

    # --------------------------------------------------------
    # Save workbook
    # --------------------------------------------------------

    workbook.save(
        file_path
    )

    print(
        "\nValidation results saved to:"
    )

    print(
        file_path
    )

    return file_path


# ============================================================
# CALCULATE VALIDATION RESULTS
# ============================================================

def calculate_validation_results(
    captured_client_weights,
    validation_dataset,
    model_builder=create_keras_model
):
    """
    Evaluate every client's trained model on the trusted
    head-client validation dataset.

    IMPORTANT:
    - Uses CPU so validation does not consume additional GPU memory.
    - Creates ONE Keras model and reuses it for all clients.
    """

    validation_results = {}

    with tf.device("/CPU:0"):

        # ----------------------------------------------------
        # Create one model and reuse it
        # ----------------------------------------------------

        evaluation_model = model_builder()

        evaluation_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # ----------------------------------------------------
        # Every federated round
        # ----------------------------------------------------

        for (
            round_idx,
            round_client_weights
        ) in enumerate(
            captured_client_weights
        ):

            round_key = (
                f"round{round_idx}weights"
            )

            validation_results[
                round_key
            ] = {}

            print(
                f"\nEvaluating clients for "
                f"{round_key}"
            )

            # ------------------------------------------------
            # Every client
            # ------------------------------------------------

            for (
                client_idx,
                client_weights
            ) in enumerate(
                round_client_weights
            ):

                client_id = (
                    f"client_{client_idx}"
                )

                # --------------------------------------------
                # Load client's trained weights
                # --------------------------------------------

                evaluation_model.set_weights(
                    [
                        weight.numpy()
                        if tf.is_tensor(weight)
                        else weight

                        for weight
                        in client_weights
                    ]
                )

                # --------------------------------------------
                # Evaluate client model
                # --------------------------------------------

                (
                    loss,
                    accuracy
                ) = evaluation_model.evaluate(
                    validation_dataset,
                    verbose=0
                )

                validation_results[
                    round_key
                ][
                    client_id
                ] = {
                    "loss": float(loss),
                    "accuracy": float(accuracy)
                }

                print(
                    f"{client_id}: "
                    f"loss={loss:.4f}, "
                    f"accuracy={accuracy:.4f}"
                )

    # ========================================================
    # SAVE RESULTS TO EXCEL
    # ========================================================

    save_validation_results_to_excel(
        validation_results
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return validation_results


# ============================================================
# EXTRACT VALIDATION ACCURACY SCORES
# ============================================================

def validation_accuracy_scores(
    validation_results
):
    """
    Extract only validation accuracy values.

    Returns:

    {
        "round0weights": {
            "client_0": 0.91,
            "client_1": 0.87,
            ...
        }
    }
    """

    scores = {}

    for (
        round_key,
        client_results
    ) in validation_results.items():

        scores[
            round_key
        ] = {}

        for (
            client_id,
            result
        ) in client_results.items():

            scores[
                round_key
            ][
                client_id
            ] = result[
                "accuracy"
            ]

    return scores