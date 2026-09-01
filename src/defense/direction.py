import numpy as np
import tensorflow as tf


def angle_between_vectors(w1, w2, eps=1e-12):

    if isinstance(w1, tf.Tensor):
        w1 = w1.numpy()

    if isinstance(w2, tf.Tensor):
        w2 = w2.numpy()

    dot = np.dot(w1, w2)

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    cosine_similarity = dot / (norm1 * norm2 + eps)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    angle_rad = np.arccos(cosine_similarity)
    return np.degrees(angle_rad)


def calculate_head_client_angles(client_weight_deltas, head_client="client_0"):
    """
    Calculate angle between the head-client update
    and every client update for every round.

    Returns:

    {
        "round0weights": {
            "client_0": angle,
            "client_1": angle,
            ...
        },

        "round1weights": {
            ...
        }
    }
    """

    results = {}
    for round_key, clients in client_weight_deltas.items():
        if head_client not in clients:
            raise ValueError(f"{head_client} missing in {round_key}")
        head_vector = clients[head_client]
        results[round_key] = {}

        for client_id, vector in clients.items():
            angle = angle_between_vectors(head_vector, vector)
            results[round_key][client_id] = angle

    return results
