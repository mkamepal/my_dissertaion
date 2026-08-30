import tensorflow as tf

def flatten_weights(weights):

    return tf.concat([tf.reshape(weight,[-1]) for weight in weights],
        axis=0
    )

def calculate_client_deltas(captured_client_weights, server_weights_before_round):

    client_weight_deltas = {}

    for (round_idx, round_client_weights) in enumerate(captured_client_weights):

        round_key = f"round{round_idx}weights"
        client_weight_deltas[round_key] = {}

        server_trainable = server_weights_before_round[round_idx].trainable
        with tf.device("/CPU:0"):
            for (client_idx, client_trainable) in enumerate(round_client_weights):

                delta_tensors = [client_w - server_w for (client_w, server_w) in zip(client_trainable, server_trainable)]
                flat_delta = tf.concat( [tf.reshape(delta,[-1]) for delta in delta_tensors], axis=0)
                client_weight_deltas[round_key][f"client_{client_idx}"] = flat_delta

    return client_weight_deltas