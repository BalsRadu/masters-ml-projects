import math
from random import random

import numpy as np
from clustering.utils import dist_euclidean


# ----------------------------------------------------------------------
#                    Fuzzy C-Means Utilities
# ----------------------------------------------------------------------
def initialize_membership_matrix(data: np.ndarray, n_clusters: int) -> list[list[float]]:
    """
    Initializes the membership matrix. Each row is a data point, and each 
    column is a cluster membership. This code normalizes random values 
    across clusters, but then forces a "crisp" membership by setting 
    the largest to 1 and others to 0.

    :param data: The dataset as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :return: Membership matrix of shape (n_samples, n_clusters).
    """
    membership = []
    for _ in range(len(data)):
        row_rand = [random() for _ in range(n_clusters)]
        s = sum(row_rand)
        row_norm = [val / s for val in row_rand]
        max_idx = np.argmax(row_norm)
        row_crisp = [1.0 if i == max_idx else 0.0 for i in range(n_clusters)]
        membership.append(row_crisp)
    return membership


def update_centroids_fuzzy(
    membership: list[list[float]],
    data: np.ndarray,
    n_clusters: int, 
    fuzz: float
) -> list[np.ndarray]:
    """
    Updates centroids based on the membership matrix (Fuzzy C-means update).

    :param membership: Membership matrix of shape (n_samples, n_clusters).
    :param data: Data points as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :param fuzz: Fuzzification parameter (typically > 1).
    :return: Updated centroids as a list of numpy arrays.
    """
    n_features = data.shape[1]
    new_centers = []
    for j in range(n_clusters):
        numerator = np.zeros(n_features)
        denominator = 0.0
        for i, point in enumerate(data):
            w = membership[i][j] ** fuzz
            numerator += w * point
            denominator += w
        # Avoid dividing by zero
        new_centers.append(numerator / denominator if denominator else numerator)
    return new_centers


def update_membership_matrix(
    membership: list[list[float]],
    centers: list[np.ndarray], 
    data: np.ndarray,
    n_clusters: int, 
    fuzz: float
) -> list[list[float]]:
    """
    Updates the membership matrix based on newly computed centroids (Fuzzy C-Means).
    
    :param membership: Current membership matrix.
    :param centers: list of cluster centroids as (n_clusters, n_features).
    :param data: The dataset as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :param fuzz: Fuzzification parameter.
    :return: Updated membership matrix of shape (n_samples, n_clusters).
    """
    ratio = 2.0 / (fuzz - 1.0)
    for i, point in enumerate(data):
        distances = [dist_euclidean(point, c) for c in centers]
        # If the point is exactly on one centroid, membership is 1 for that centroid
        # and 0 for all others
        if any(d == 0 for d in distances):
            zero_idx = distances.index(0)
            membership[i] = [1.0 if z == zero_idx else 0.0 for z in range(n_clusters)]
        else:
            membership[i] = [
                1.0 / sum(
                    ((distances[j] / (distances[q] if distances[q] != 0 else 1e-12)) ** ratio)
                    for q in range(n_clusters)
                )
                for j in range(n_clusters)
            ]
    return membership


def get_clusters_and_labels(
    membership: list[list[float]], 
    data: np.ndarray, 
    n_clusters: int
) -> tuple[list[list[np.ndarray]], list[int]]:
    """
    Converts membership matrix to "hard" clusters and labels by 
    taking the argmax membership for each point.

    :param membership: Membership matrix of shape (n_samples, n_clusters).
    :param data: The dataset as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :return: (clusters, labels)
             clusters -> list of n_clusters lists of points
             labels   -> list of length n_samples containing cluster indices
    """
    clusters = [[] for _ in range(n_clusters)]
    labels = [-1] * len(data)
    for i, point in enumerate(data):
        max_idx = np.argmax(membership[i])
        clusters[max_idx].append(point)
        labels[i] = max_idx
    return clusters, labels


# ----------------------------------------------------------------------
#                           Fuzzy C-Means Clustering
# ----------------------------------------------------------------------
class FuzzyCMeans:
    def __init__(
        self, 
        n_clusters: int = 3, 
        max_iter: int = 100, 
        tolerance: float = 0.001, 
        fuzzification: float = 2.0
    ) -> None:
        """
        Fuzzy C-Means clustering.
        
        :param n_clusters: Number of clusters.
        :param max_iter: Maximum iterations.
        :param tolerance: Tolerance for centroid shifts.
        :param fuzzification: Fuzzification parameter (m > 1).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fuzz = fuzzification
        self.centers: list[np.ndarray] = []

    def run(
        self, 
        data: np.ndarray
    ) -> tuple[
        list[list[np.ndarray]], 
        list[np.ndarray], 
        list[int], 
        list[list[float]]
    ]:
        """
        Executes the Fuzzy C-Means clustering on 'data'.

        :param data: Dataset as (n_samples, n_features).
        :return: (clusters, centers, labels, membership)
                 clusters   -> list of lists of data points
                 centers    -> list of cluster centers
                 labels     -> crisp label for each point
                 membership -> the final membership matrix
        """
        # Initialize membership matrix
        membership = initialize_membership_matrix(data, self.n_clusters)

        for iteration in range(self.max_iter):
            prev_centers = self.centers.copy() if iteration > 0 else None

            # Update centroids
            self.centers = update_centroids_fuzzy(membership, data, self.n_clusters, self.fuzz)
            # Update membership
            membership = update_membership_matrix(membership, self.centers, data, self.n_clusters, self.fuzz)
            # Convert to "hard" clusters
            clusters, labels = get_clusters_and_labels(membership, data, self.n_clusters)

            # Check convergence if not the first iteration
            if iteration > 0 and prev_centers is not None:
                shifts = [
                    dist_euclidean(prev_centers[i], self.centers[i]) 
                    for i in range(self.n_clusters)
                ]
                if all(shift <= self.tolerance for shift in shifts):
                    print(f"FuzzyCMeans converged after iteration: {iteration + 1}")
                    return clusters, self.centers, labels, membership

        print(f"FuzzyCMeans ended after max_iter = {self.max_iter}")
        return clusters, self.centers, labels, membership