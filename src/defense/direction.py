import numpy as np
import tensorflow as tf


def vector_direction(w, eps=1e-12):
    """Return the L2-normalized direction of a client-weight vector."""
    if isinstance(w, tf.Tensor):
        w = w.numpy()

    mag = np.linalg.norm(w)
    return w / (mag + eps)


def angle_between_vectors(w1, w2, eps=1e-12):

    if isinstance(w1, tf.Tensor):
        w1 = w1.numpy()

    if isinstance(w2, tf.Tensor):
        w2 = w2.numpy()

    dot = np.dot(w1, w2)

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    cos_sim = dot / (norm1 * norm2 + eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    angle_rad = np.arccos(cos_sim)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_head_client_angles(client_weight_vectors, head_client="client_0"):
    """
    Calculate the angle from the head client's flattened vector to every
    other client's flattened vector for every round.

    This is the client_0-only form of the notebook's pairwise comparison.
    The head client is retained with an exact zero-degree angle because the
    trust-score combiner requires a direction score for every client.

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
    for round_key, clients in client_weight_vectors.items():
        if head_client not in clients:
            raise ValueError(f"{head_client} missing in {round_key}")

        head_vector = clients[head_client]
        results[round_key] = {head_client: 0.0}

        for client_id, vector in clients.items():
            if client_id == head_client:
                continue

            angle = angle_between_vectors(head_vector, vector)
            results[round_key][client_id] = angle

    return results
