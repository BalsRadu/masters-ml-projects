import contextlib
import math
import os
import sys
from random import random

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------
#                    Context Manager
# ----------------------------------------------------------------------
@contextlib.contextmanager
def suppress_output():
    """
    A context manager to suppress standard output and standard error.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ----------------------------------------------------------------------
#                           Data Loading
# ----------------------------------------------------------------------

def load_data(filename: str) -> np.ndarray:
    """
    Reads a dataset from a .csv file. For 'Dataset2.csv', it reads
    all columns; for other files, it discards the last column.
    
    :param filename: Path to the .csv file.
    :return: A numpy array of shape (n_samples, n_features).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_rows = [
        [float(x) for x in line.strip().split(',')[:-1]]
        if filename != 'Dataset2.csv'
        else [float(x) for x in line.strip().split(',')]
        for line in lines
    ]
    return np.array(data_rows)

# ----------------------------------------------------------------------
#                           Distance Function
# ----------------------------------------------------------------------

def dist_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes Euclidean distance between two points (1D vectors).
    
    :param a: First point as a numpy array.
    :param b: Second point as a numpy array.
    :return: Euclidean distance between a and b.
    """
    return np.sqrt(np.sum((a - b) ** 2))

# ----------------------------------------------------------------------
#                          Plotting Functions
# ----------------------------------------------------------------------

def plot_iris_clusters(
    clusters: list[list[np.ndarray]], 
    centers: list[np.ndarray]
) -> None:
    """
    Plots clusters specifically for the Iris dataset example. 
    Assumes 3 clusters.

    :param clusters: A list of 3 lists, each containing points in that cluster.
    :param centers: A list of 3 centroids.
    """
    # Define colors for the three clusters
    colors = ['red', 'green', 'blue']
    
    # Create a scatter plot for each cluster
    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f"Cluster {i+1}")
    
    # Plot the centroids
    centroids_array = np.array(centers)
    plt.scatter(centroids_array[:, 0], centroids_array[:, 1], 
                c='black', marker='x', s=100, label="Centroids")
    
    # Set plot labels and title
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.title("Iris Dataset Clusters")
    plt.legend()
    plt.show()


def plot_clusters(
    clusters: list[list[np.ndarray]],
    centers: list[np.ndarray],
    feature_names: list[str],
    title: str
) -> None:
    """
    Plots clusters and their centroids. Assumes 2D data with 3 clusters.
    
    :param clusters: A list of clusters, where each cluster is a list of points.
    :param centers: A list of centroids.
    :param feature_names: Feature names for the axes.
    :param dataset_name: Name of the dataset (e.g., "Iris") to use in the title.
    """
    # Define colors for clusters
    colors = ['red', 'green', 'blue']
    
    # Plot clusters
    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    c=colors[i], alpha=0.8, label=f"Cluster {i+1}")
    
    # Plot centroids
    centroids_array = np.array(centers)
    plt.scatter(centroids_array[:, 0], centroids_array[:, 1], 
                c='black', marker='x', s=100, label="Centroids")
    
    # Add labels, title, and legend
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()
# ----------------------------------------------------------------------
#                          Data Manipulation
# ----------------------------------------------------------------------
def reduce_to_2d(data: np.ndarray) -> np.ndarray:
    """
    Reduces the dataset to 2D using PCA.
    
    :param data: Dataset as (n_samples, n_features).
    :return: Dataset reduced to 2D.
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

def remove_outliers(df, threshold=1.5):
    """
    Removes rows that contain outliers in any column based on the IQR method.

    :param df: DataFrame to process
    :param threshold: Multiplier for the IQR to define outlier bounds
    :return: DataFrame with outliers removed
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# ----------------------------------------------------------------------
#                          Metric Functions
# ----------------------------------------------------------------------

def calculate_sse(
    clusters: list[list[np.ndarray]], 
    centroids: list[np.ndarray], 
    n_clusters: int
) -> float:
    """
    Calculates the sum of squared errors (SSE) given clusters and their respective centers.

    :param clusters: A list of clusters, where each cluster is a list of points (np.ndarray).
    :param centroids: The corresponding cluster centroids.
    :param n_clusters: Number of clusters.
    :return: The SSE value (float).
    """
    return sum(
        np.sum((dist_euclidean(point, centroids[k]) ** 2) for point in clusters[k])
        for k in range(n_clusters)
    )

def calculate_sse_fuzzy(
    data: np.ndarray,
    centroids: list[np.ndarray],
    n_clusters: int,
    member_mat: np.ndarray,
    fuzziness: float = 2.0
) -> float:
    """
    Computes the fuzzy SSE for a set of clusters in Fuzzy C-Means.

    :param member_mat: Membership matrix (shape: [n_samples, n_clusters]), 
                       indicating degree of belonging for each point to each cluster.
    :param data: Dataset as a numpy array (shape: [n_samples, n_features]).
    :param centroids: List of cluster centroids (np.ndarray).
    :param n_clusters: Number of clusters.
    :param fuzziness: Fuzziness parameter (typically m=2.0).
    :return: Total fuzzy SSE value.
    """
    return sum(
        np.sum(member_mat[i][k] ** fuzziness * dist_euclidean(data[i], centroids[k]) ** 2 for i in range(len(data)))
        for k in range(n_clusters)
    )



def find_elbow_point(sse_values, ks, plot=False):
    """
    Finds the elbow point in the SSE values using the KneeLocator.
    
    :param sse_values: List or array of SSE values.
    :param ks: List or array of corresponding cluster numbers.
    :param plot: Whether to plot the SSE curve and highlight the elbow point.
    :return: Optimal number of clusters (k) and the index of the elbow.
    """
    kn = KneeLocator(ks, sse_values, curve="convex", direction="decreasing")
    optimal_k = kn.knee

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(ks, sse_values, marker="o")
        plt.xticks(ks)
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("Elbow Method for Optimal k")
        if optimal_k:
            plt.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
            plt.legend()
        plt.show()

    return optimal_k

def compute_labelling_similarity(cluster_labels, true_labels):
    """
    Computes the similarity between the labels assigned by the clustering algorithm
    and the true labels.
    """
    matches = np.sum(cluster_labels == true_labels)
    return matches / len(true_labels)
    return matches / len(true_labels)
