import numpy as np
import tensorflow as tf


def vector_magnitude(vector):
    if isinstance(vector, tf.Tensor):
        vector = vector.numpy()

    return np.linalg.norm(vector, ord=2)


def calculate_magnitudes(client_weight_deltas):
    """
    Calculate the L2 magnitude of each client's flattened weight delta.
    Returns:
    {
        "round0weights": {
            "client_0": magnitude,
            "client_1": magnitude,
            ...
        },

        "round1weights": {
            ...
        }
    }
    """
    round_magnitudes = {}
    for round_key, clients in client_weight_deltas.items():
        round_magnitudes[round_key] = {}
        for client_id, vector in clients.items():
            round_magnitudes[round_key][client_id] = vector_magnitude(vector)
    return round_magnitudes
