import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import src.config.constants as constants

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# ============================================================
# HDBSCAN FOR ONE ROUND
# ============================================================


def hdbscan_one_round(
    client_delta_dict, pca_var=0.95, min_cluster_size=2, min_samples=1
):
    """
    Run HDBSCAN clustering for one federated round.

    Parameters
    ----------
    client_delta_dict : dict
        Dictionary containing client update vectors.

        Example:
        {
            "client_0": vector,
            "client_1": vector,
            ...
        }

    pca_var : float
        Percentage of variance preserved by PCA.

    min_cluster_size : int
        Minimum number of samples required to form a cluster.

    min_samples : int
        HDBSCAN min_samples parameter.

    Returns
    -------
    labels : np.ndarray
        Cluster labels.
        -1 means noise/outlier.

    probabilities : np.ndarray
        HDBSCAN membership confidence for each client.

    Z : np.ndarray
        PCA-transformed client vectors.

    client_keys : list
        Ordered client IDs.
    """

    # --------------------------------------------------------
    # Sort clients numerically
    # --------------------------------------------------------

    client_keys = sorted(
        client_delta_dict.keys(), key=lambda value: int(value.split("_")[1])
    )

    # --------------------------------------------------------
    # Convert client vectors to NumPy
    # --------------------------------------------------------

    X = np.stack(
        [
            (
                client_delta_dict[client].numpy()
                if hasattr(client_delta_dict[client], "numpy")
                else client_delta_dict[client]
            )
            for client in client_keys
        ],
        axis=0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Center across clients
    # --------------------------------------------------------

    X = X - X.mean(axis=0, keepdims=True)

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    X_std = X.std(axis=0, keepdims=True)

    X_std[X_std == 0] = 1.0

    X_norm = X / X_std

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(n_components=pca_var, random_state=42)

    Z = pca.fit_transform(X_norm)

    # --------------------------------------------------------
    # Normalize PCA features
    # --------------------------------------------------------

    Z_cos = normalize(Z, norm="l2")

    # --------------------------------------------------------
    # HDBSCAN
    # --------------------------------------------------------

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(Z_cos)

    probabilities = clusterer.probabilities_

    return (labels, probabilities, Z, client_keys)


# ============================================================
# PRINT RESULTS
# ============================================================


def print_hdbscan_results(hdbscan_results):
    """
    Print cluster assignment and probability
    for every client in every round.
    """

    for round_key, result in hdbscan_results.items():

        print("\n" + "=" * 60)

        print(round_key)

        print("=" * 60)

        labels = result["labels"]

        probabilities = result["probabilities"]

        client_keys = result["client_keys"]

        for client_id, label, probability in zip(client_keys, labels, probabilities):

            print(
                f"{client_id:12s} "
                f"-> Cluster: {int(label):2d} | "
                f"Probability: {float(probability):.4f}"
            )


# ============================================================
# PLOT ONE ROUND
# ============================================================


def plot_hdbscan_round(
    round_key, labels, probabilities, features, client_keys, output_dir, show_plot=False
):
    """
    Save one HDBSCAN graph for one federated round.
    """

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Use first two PCA components
    # --------------------------------------------------------

    if features.shape[1] >= 2:

        x = features[:, 0]
        y = features[:, 1]

    else:

        x = features[:, 0]

        y = np.zeros_like(features[:, 0])

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    unique_labels = sorted(set(labels))

    # --------------------------------------------------------
    # Plot each cluster
    # --------------------------------------------------------

    for cluster_label in unique_labels:

        indices = labels == cluster_label

        if cluster_label == -1:

            # Noise / outlier clients
            plt.scatter(
                x[indices],
                y[indices],
                marker="x",
                s=150,
                linewidths=2,
                label="Noise / Outlier",
            )

        else:
            # Larger marker means higher HDBSCAN confidence
            point_sizes = 120 + 300 * probabilities[indices]
            plt.scatter(
                x[indices],
                y[indices],
                s=point_sizes,
                marker="o",
                label=f"Cluster {cluster_label}",
            )

    for idx, client_id in enumerate(client_keys):
        plt.annotate(
            client_id,
            (x[idx], y[idx]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    number_of_clusters = len([label for label in unique_labels if label != -1])

    number_of_noise = int(np.sum(labels == -1))

    plt.title(
        f"{round_key} - HDBSCAN Clustering\n"
        f"Clusters: {number_of_clusters} | "
        f"Noise: {number_of_noise}"
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    file_path = os.path.join(output_dir, f"{round_key}_hdbscan.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return file_path


def save_all_hdbscan_plots(hdbscan_results, output_dir, show_plots=False):
    """
    Save one HDBSCAN graph for every federated round.
    """

    plot_dir = os.path.join(output_dir, "hdbscan_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for round_key, result in hdbscan_results.items():

        plot_hdbscan_round(
            round_key=round_key,
            labels=result["labels"],
            probabilities=result["probabilities"],
            features=result["features"],
            client_keys=result["client_keys"],
            output_dir=plot_dir,
            show_plot=show_plots,
        )

    print("\nHDBSCAN graphs saved to:")
    print(plot_dir)
    return plot_dir


def calculate_hdbscan_results(
    client_weight_deltas,
    pca_var=0.95,
    min_cluster_size=2,
    min_samples=1,
    print_results=True,
):
    """
    Main HDBSCAN function.

    This is the only function that needs to be called
    from run_tff.py.

    It:
        1. Calculates HDBSCAN for every round.
        2. Prints results.
        3. Returns the HDBSCAN result dictionary.
    """

    results = {}

    for round_key, clients in client_weight_deltas.items():
        (labels, probabilities, Z, client_keys) = hdbscan_one_round(
            clients,
            pca_var=pca_var,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        results[round_key] = {
            "labels": labels,
            "probabilities": probabilities,
            "features": Z,
            "client_keys": client_keys,
        }

    if print_results:
        print_hdbscan_results(results)

    return results
