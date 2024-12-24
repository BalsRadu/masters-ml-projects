import math
from random import random

import numpy as np
from clustering.utils import dist_euclidean


# ----------------------------------------------------------------------
#                    K-Means Utilities
# ----------------------------------------------------------------------
# TODO: Remove the random initialization method and implement the k-means++
def init_centroids_random(data: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """
    Selects random points from the dataset to be centroids.
    
    :param data: The dataset as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :return: list of centroids, each centroid is shape (n_features,).
    """
    return [data[np.random.choice(len(data))] for _ in range(n_clusters)]


def init_centroids_pp(data: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """
    Initializes centroids using the k-means++ method.

    :param data: The dataset as (n_samples, n_features).
    :param n_clusters: Number of clusters.
    :return: list of centroids, each centroid is shape (n_features,).
    """
    centers: list[np.ndarray] = []
    # 1. Choose one center uniformly at random
    idx = np.random.choice(len(data))
    centers.append(data[idx])

    for _ in range(n_clusters - 1):
        # Compute distance from each point to the nearest center
        dists_sq = [
            np.min([dist_euclidean(point, c) for c in centers])**2 
            for point in data
        ]
        dists_sq = np.array(dists_sq) / np.sum(dists_sq)

        # Choose new center according to weighted probability
        new_idx = np.random.choice(len(data), p=dists_sq)
        # If duplicate, keep picking until distinct
        while any(np.allclose(data[new_idx], c) for c in centers):
            new_idx = np.random.choice(len(data), p=dists_sq)
        
        centers.append(data[new_idx])
    
    return centers
# ----------------------------------------------------------------------
#                            K-Means Clustering
# ----------------------------------------------------------------------
class KMeans:
    def __init__(
        self, 
        n_clusters: int = 3, 
        max_iter: int = 100, 
        tolerance: float = 0.001
    ) -> None:
        """
        KMeans clustering implementation.
        
        :param n_clusters: Number of clusters.
        :param max_iter: Maximum number of iterations.
        :param tolerance: Stop if centroid shift is below this threshold.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centers: list[np.ndarray] = []

    def run(
        self, 
        data: np.ndarray
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[int]]:
        """
        Executes the K-Means clustering on 'data'.

        :param data: The dataset as (n_samples, n_features).
        :return: (clusters, centers, labels) 
                 clusters -> list of lists of data points
                 centers -> list of cluster centers
                 labels -> integer labels for each point
        """
        # Initialize centroids (use plus-plus method)
        # We will not use the random initialization because
        # it can lead to poor convergence in some cases.
        self.centers = init_centroids_pp(data, self.n_clusters)
        labels = [-1] * len(data)
        iteration = 0

        for _ in range(self.max_iter):
            iteration += 1
            # Assign points to nearest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for i, point in enumerate(data):
                dists = [dist_euclidean(point, c) for c in self.centers]
                c_idx = np.argmin(dists)
                clusters[c_idx].append(point)
                labels[i] = c_idx
            
            prev_centers = self.centers.copy()
            # Recompute centroids
            self.centers = [
                np.mean(cluster, axis=0) if len(cluster) else data[np.random.choice(len(data))]
                for cluster in clusters
            ]

            # Check centroid shifts
            shifts = [
                dist_euclidean(prev_centers[i], self.centers[i]) 
                for i in range(self.n_clusters)
            ]
            if all(shift <= self.tolerance for shift in shifts):
                print(f"KMeans converged after iteration: {iteration}")
                return clusters, self.centers, labels

        print(f"KMeans ended after max_iter = {self.max_iter}")
        return clusters, self.centers, labels