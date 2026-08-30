import os
import numpy as np
import tensorflow as tf

from openpyxl import Workbook
from src.config import constants

def angle_between_vectors(
        w1,
        w2,
        eps=1e-12
):

    if isinstance(w1, tf.Tensor):
        w1 = w1.numpy()

    if isinstance(w2, tf.Tensor):
        w2 = w2.numpy()

    dot = np.dot(w1, w2)

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    cosine_similarity = (dot / (norm1 * norm2 + eps))
    cosine_similarity = np.clip(cosine_similarity, -1.0,1.0)
    angle_rad = np.arccos(cosine_similarity)
    return np.degrees(angle_rad)

def save_directions_to_excel(direction_results, head_client="client_0"):
    """
    Save direction/angle values for every client
    in every federated round.

    Columns:
        Round
        Head Client
        Client ID
        Angle (Degrees)
    """

    output_dir = constants.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"directions_clients_{constants.NUM_CLIENTS}_malicious_{constants.NUM_MALICIOUS}_rounds_{constants.FEDERATED_ROUNDS}.xlsx"

    file_path = os.path.join(output_dir, file_name)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Direction Results"
    worksheet.append(["Round", "Head Client", "Client ID", "Angle (Degrees)"])

    for (round_idx,(round_key, clients)) in enumerate(direction_results.items(), start=1):
        client_keys = sorted(clients.keys(), key=lambda value: int(value.split("_")[1]))
        for client_id in client_keys:
            angle = clients[client_id]
            worksheet.append([round_idx, head_client, client_id, float(angle)])

    worksheet.column_dimensions["A"].width = 12
    worksheet.column_dimensions["B"].width = 18
    worksheet.column_dimensions["C"].width = 18
    worksheet.column_dimensions["D"].width = 22
    workbook.save(file_path)
    print("\nDirection results saved to:")

    print(file_path)


def calculate_head_client_angles(
        client_weight_deltas,
        head_client="client_0"
):
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
    for (round_key, clients) in client_weight_deltas.items():
        if head_client not in clients:
            raise ValueError(f"{head_client} missing in {round_key}")
        head_vector = clients[head_client]
        results[round_key] = {}

        for (client_id, vector) in clients.items():
            angle = angle_between_vectors(head_vector, vector)
            results[round_key][client_id] = angle

    save_directions_to_excel(direction_results=results, head_client=head_client)
    return results