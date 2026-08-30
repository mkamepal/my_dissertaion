import os
import numpy as np

from openpyxl import Workbook

from src.config import constants


# ============================================================
# CLUSTERING SCORE
# ============================================================

def calculate_clustering_scores(
    hdbscan_results,
    head_client="client_0"
):
    """
    Binary clustering score.

    C_i = 1
        if client belongs to the same HDBSCAN cluster
        as the head client.

    C_i = 0
        otherwise.

    If the head client is classified as noise (-1),
    only the head client receives score 1.
    """

    clustering_scores = {}

    for round_key, result in hdbscan_results.items():

        clustering_scores[round_key] = {}

        labels = result["labels"]
        client_keys = result["client_keys"]

        if head_client not in client_keys:
            raise ValueError(
                f"{head_client} missing in {round_key}"
            )

        head_index = client_keys.index(
            head_client
        )

        head_cluster = labels[
            head_index
        ]

        # ----------------------------------------------------
        # Head client classified as noise
        # ----------------------------------------------------

        if head_cluster == -1:

            for client_id in client_keys:

                clustering_scores[
                    round_key
                ][
                    client_id
                ] = (
                    1
                    if client_id == head_client
                    else 0
                )

            continue

        # ----------------------------------------------------
        # Normal cluster comparison
        # ----------------------------------------------------

        for client_id, label in zip(
            client_keys,
            labels
        ):

            if label == head_cluster:

                clustering_scores[
                    round_key
                ][
                    client_id
                ] = 1

            else:

                clustering_scores[
                    round_key
                ][
                    client_id
                ] = 0

    return clustering_scores


# ============================================================
# DIRECTION SCORE
# ============================================================

def calculate_direction_scores(
    head_angles,
    threshold
):
    """
    Binary direction score.

    D_i = 1 if angle <= threshold
    D_i = 0 otherwise
    """

    direction_scores = {}

    for round_key, clients in head_angles.items():

        direction_scores[
            round_key
        ] = {}

        for client_id, angle in clients.items():

            if angle <= threshold:

                score = 1

            else:

                score = 0

            direction_scores[
                round_key
            ][
                client_id
            ] = score

    return direction_scores


# ============================================================
# MAGNITUDE SCORE
# ============================================================

def calculate_magnitude_scores(
    magnitudes,
    head_client="client_0",
    threshold=0.20,
    eps=1e-12
):
    """
    Binary magnitude score.

    Uses relative magnitude difference:

        |m_i - m_h|
        -----------
          |m_h|

    M_i = 1 if relative difference <= threshold
    M_i = 0 otherwise

    Example:
        threshold = 0.20

    means the client magnitude can differ by up to
    20% from the head-client magnitude.
    """

    magnitude_scores = {}

    for round_key, clients in magnitudes.items():

        magnitude_scores[
            round_key
        ] = {}

        if head_client not in clients:

            raise ValueError(
                f"{head_client} missing in {round_key}"
            )

        head_magnitude = float(
            clients[
                head_client
            ]
        )

        for client_id, magnitude in clients.items():

            magnitude = float(
                magnitude
            )

            relative_difference = (
                abs(
                    magnitude
                    - head_magnitude
                )
                /
                (
                    abs(
                        head_magnitude
                    )
                    + eps
                )
            )

            if relative_difference <= threshold:

                score = 1

            else:

                score = 0

            magnitude_scores[
                round_key
            ][
                client_id
            ] = score

    return magnitude_scores


# ============================================================
# VALIDATION SCORE
# ============================================================

def calculate_validation_scores(
    validation_results,
    head_client="client_0",
    accuracy_tolerance=0.10
):
    """
    Binary validation score.

    Uses validation accuracy relative to the head client.

    V_i = 1 if:

        client_accuracy
        >=
        head_accuracy - accuracy_tolerance

    Otherwise:

        V_i = 0

    Example:

        head accuracy = 0.90
        tolerance     = 0.10

        minimum acceptable accuracy = 0.80
    """

    validation_scores = {}

    for round_key, clients in validation_results.items():

        validation_scores[
            round_key
        ] = {}

        if head_client not in clients:

            raise ValueError(
                f"{head_client} missing in {round_key}"
            )

        head_accuracy = float(
            clients[
                head_client
            ][
                "accuracy"
            ]
        )

        minimum_accuracy = (
            head_accuracy
            - accuracy_tolerance
        )

        for client_id, result in clients.items():

            client_accuracy = float(
                result[
                    "accuracy"
                ]
            )

            if client_accuracy >= minimum_accuracy:

                score = 1

            else:

                score = 0

            validation_scores[
                round_key
            ][
                client_id
            ] = score

    return validation_scores


# ============================================================
# SAVE TRUST RESULTS TO EXCEL
# ============================================================

def save_trust_scores_to_excel(
    trust_results
):
    """
    Save all binary component scores and final trust score.

    Columns:

        Round
        Client ID
        Clustering Score
        Direction Score
        Magnitude Score
        Validation Score
        Trust Score
    """

    output_dir = constants.RESULTS_DIR

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    file_name = (
        f"trust_scores"
        f"_clients_{constants.NUM_CLIENTS}"
        f"_malicious_{constants.NUM_MALICIOUS}"
        f".xlsx"
    )

    file_path = os.path.join(
        output_dir,
        file_name
    )

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = (
        "Trust Scores"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    worksheet.append([
        "Round",
        "Client ID",
        "Clustering Score",
        "Direction Score",
        "Magnitude Score",
        "Validation Score",
        "Trust Score"
    ])

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    for (
        round_idx,
        (
            round_key,
            clients
        )
    ) in enumerate(
        trust_results.items(),
        start=1
    ):

        client_keys = sorted(
            clients.keys(),
            key=lambda value: int(
                value.split("_")[1]
            )
        )

        for client_id in client_keys:

            result = clients[
                client_id
            ]

            worksheet.append([
                round_idx,
                client_id,

                int(
                    result[
                        "clustering_score"
                    ]
                ),

                int(
                    result[
                        "direction_score"
                    ]
                ),

                int(
                    result[
                        "magnitude_score"
                    ]
                ),

                int(
                    result[
                        "validation_score"
                    ]
                ),

                float(
                    result[
                        "trust_score"
                    ]
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
    ].width = 20

    worksheet.column_dimensions[
        "D"
    ].width = 20

    worksheet.column_dimensions[
        "E"
    ].width = 20

    worksheet.column_dimensions[
        "F"
    ].width = 20

    worksheet.column_dimensions[
        "G"
    ].width = 18

    # --------------------------------------------------------
    # Save workbook
    # --------------------------------------------------------

    workbook.save(
        file_path
    )

    print(
        "\nTrust scores saved to:"
    )

    print(
        file_path
    )

    return file_path


# ============================================================
# FINAL COMBINED TRUST SCORE
# ============================================================

def calculate_trust_scores(
    hdbscan_results,
    head_angles,
    magnitudes,
    validation_results,
    head_client="client_0"
):
    """
    Calculate the final combined trust score.

    Each component is binary:

        C_i ∈ {0,1}
        D_i ∈ {0,1}
        M_i ∈ {0,1}
        V_i ∈ {0,1}

    Final trust:

        T_i =
            w_c * C_i
            +
            w_d * D_i
            +
            w_m * M_i
            +
            w_v * V_i
    """

    # ========================================================
    # GET PARAMETERS FROM CONSTANTS
    # ========================================================

    w_c = constants.TRUST_WEIGHT_CLUSTER
    w_d = constants.TRUST_WEIGHT_DIRECTION
    w_m = constants.TRUST_WEIGHT_MAGNITUDE
    w_v = constants.TRUST_WEIGHT_VALIDATION

    direction_threshold = (
        constants.DIRECTION_THRESHOLD
    )

    magnitude_threshold = (
        constants.MAGNITUDE_THRESHOLD
    )

    validation_tolerance = (
        constants.VALIDATION_ACCURACY_TOLERANCE
    )

    # ========================================================
    # CHECK WEIGHTS
    # ========================================================

    total_weight = (
        w_c
        + w_d
        + w_m
        + w_v
    )

    if not np.isclose(
        total_weight,
        1.0
    ):

        raise ValueError(
            "Trust weights must sum to 1.0. "
            f"Current total = {total_weight}"
        )

    # ========================================================
    # CALCULATE INDIVIDUAL BINARY SCORES
    # ========================================================

    clustering_scores = (
        calculate_clustering_scores(
            hdbscan_results,
            head_client=head_client
        )
    )

    direction_scores = (
        calculate_direction_scores(
            head_angles,
            threshold=direction_threshold
        )
    )

    magnitude_scores = (
        calculate_magnitude_scores(
            magnitudes,
            head_client=head_client,
            threshold=magnitude_threshold
        )
    )

    validation_scores = (
        calculate_validation_scores(
            validation_results,
            head_client=head_client,
            accuracy_tolerance=
                validation_tolerance
        )
    )

    # ========================================================
    # COMBINE SCORES
    # ========================================================

    trust_results = {}

    for (
        round_key,
        cluster_clients
    ) in clustering_scores.items():

        trust_results[
            round_key
        ] = {}

        for client_id in cluster_clients.keys():

            c = clustering_scores[
                round_key
            ][
                client_id
            ]

            d = direction_scores[
                round_key
            ][
                client_id
            ]

            m = magnitude_scores[
                round_key
            ][
                client_id
            ]

            v = validation_scores[
                round_key
            ][
                client_id
            ]

            # ------------------------------------------------
            # Weighted combined trust
            # ------------------------------------------------

            trust = (
                w_c * c
                + w_d * d
                + w_m * m
                + w_v * v
            )

            trust_results[
                round_key
            ][
                client_id
            ] = {

                "clustering_score":
                    c,

                "direction_score":
                    d,

                "magnitude_score":
                    m,

                "validation_score":
                    v,

                "trust_score":
                    float(
                        trust
                    )
            }

    # ========================================================
    # SAVE TO EXCEL
    # ========================================================

    save_trust_scores_to_excel(
        trust_results
    )

    # ========================================================
    # RETURN
    # ========================================================

    return trust_results