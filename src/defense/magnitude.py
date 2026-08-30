import os
import numpy as np
import tensorflow as tf
import src.config.constants as constants

from openpyxl import Workbook

def vector_magnitude(vector):
    if isinstance(vector,tf.Tensor):
        vector = vector.numpy()

    return np.linalg.norm(vector,ord=2)

def save_magnitudes_to_excel(
    round_magnitudes,
    total_clients,
    total_malicious,
    output_dir=constants.RESULTS_DIR
):
    """
    Save client magnitudes for all rounds into one Excel file.

    Excel structure:

    Round | Client ID | Magnitude
    """

    os.makedirs(output_dir,exist_ok=True)
    file_name = f"magnitudes_clients_{total_clients}_malicious_{total_malicious}_rounds_{constants.FEDERATED_ROUNDS}.xlsx"
    file_path = os.path.join(output_dir ,file_name)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Magnitudes"
    # Header
    worksheet.append(["Round","Client ID","Magnitude"])

    for round_idx, (round_key,client_values) in enumerate(round_magnitudes.items(),start=1):
        client_ids = sorted(client_values.keys(),key=lambda client_id: int(client_id.split("_")[1]))
        for client_id in client_ids:
            magnitude = client_values[client_id]
            worksheet.append([round_idx,client_id,float(magnitude)])

    # Column widths
    worksheet.column_dimensions["A"].width = 12
    worksheet.column_dimensions["B"].width = 18
    worksheet.column_dimensions["C"].width = 22
    workbook.save(file_path)
    print(f"Magnitude results saved to: {file_path}")

def calculate_magnitudes(client_weight_deltas):
    """
        Calculate magnitude for every client in every round.
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
    for (round_key, clients) in client_weight_deltas.items():
        round_magnitudes[round_key] = {}
        for (client_id,vector) in clients.items():
            round_magnitudes[round_key][client_id] = vector_magnitude(vector)
    save_magnitudes_to_excel(round_magnitudes=round_magnitudes,
                             total_clients=constants.NUM_CLIENTS,
                             total_malicious=constants.NUM_MALICIOUS,
                             output_dir=constants.RESULTS_DIR
                             )
    return round_magnitudes

