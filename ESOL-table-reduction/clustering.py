import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness


def calculate_trustworthiness(X_original, X_embedded, n_neighbors=5):
    """
    Calculate the trustworthiness of a dimensionality reduction

    Parameters:
    X_original: Data in original feature space
    X_embedded: Data after dimensionality reduction
    n_neighbors: Number of neighbors to consider

    Returns:
    trust: Trustworthiness score (0-1, higher is better)
    """
    # Use scikit-learn's implementation
    trust = trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)
    return trust


def calculate_continuity(X_original, X_embedded, n_neighbors=5):
    """
    Calculate the continuity of a dimensionality reduction

    Parameters:
    X_original: Data in original feature space
    X_embedded: Data after dimensionality reduction
    n_neighbors: Number of neighbors to consider

    Returns:
    continuity: Continuity score (0-1, higher is better)
    """
    # Calculate k-nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)

    # Calculate k-nearest neighbors in embedded space
    nbrs_embed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded)
    _, indices_embed = nbrs_embed.kneighbors(X_embedded)

    # Calculate continuity
    n_samples = X_original.shape[0]
    continuity_sum = 0.0

    for i in range(n_samples):
        # Get the nearest neighbors in original space (excluding the point itself)
        orig_neighbors = set(indices_orig[i][1:])

        # Get the nearest neighbors in embedded space
        embed_neighbors = set(indices_embed[i][1:])

        # Calculate the points that are in the original neighborhood but not in the embedded one
        missing_neighbors = orig_neighbors - embed_neighbors

        # Calculate the rank violations
        violation_sum = 0
        for j in missing_neighbors:
            # Find the rank of j in the embedded space
            rank_j_embed = np.where(indices_embed[i] == j)[0]
            if len(rank_j_embed) == 0:  # j is not in the k nearest neighbors in embedded space
                # Assign maximum possible rank
                rank_j_embed = n_samples
            else:
                rank_j_embed = rank_j_embed[0]

            # Rank violation (r_ij - K)
            violation_sum += (rank_j_embed - n_neighbors)

        # Add to the continuity sum
        continuity_sum += violation_sum

    # Normalize and convert to a score between 0 and 1
    normalization = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    continuity = 1.0 - normalization * continuity_sum

    return continuity


def evaluate_embedding(X_original, X_embedded, solubility_values=None, solubility_bins=None):
    """
    Evaluate quality of dimensionality reduction results

    Parameters:
    X_original: Data in original feature space
    X_embedded: Data after dimensionality reduction
    solubility_values: Continuous solubility values (for visualization)
    solubility_bins: Discretized solubility categories (for evaluating clustering quality)

    Returns:
    metrics: Dictionary containing various evaluation metrics
    """
    metrics = {}

    # 1. Silhouette coefficient (using discretized solubility category labels)
    if solubility_bins is not None and len(np.unique(solubility_bins)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_embedded, solubility_bins)
            print(f"Silhouette coefficient using discretized solubility categories: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette coefficient: {str(e)}")
            metrics['silhouette_score'] = None

    # 2. Neighbor preservation rate
    k = min(20, len(X_original) - 1)  # Number of neighbors, not exceeding number of samples minus 1

    # Calculate k nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_original)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_original)

    # Calculate k nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    distances_embedded, indices_embedded = nbrs_embedded.kneighbors(X_embedded)

    # Calculate neighbor preservation rate
    preserved_neighbors = 0
    total_neighbors = 0

    for i in range(len(X_original)):
        # Skip first neighbor (sample itself)
        orig_neighbors = set(indices_orig[i][1:])
        embedded_neighbors = set(indices_embedded[i][1:])
        preserved_neighbors += len(orig_neighbors.intersection(embedded_neighbors))
        total_neighbors += k

    metrics['neighbor_preservation'] = preserved_neighbors / total_neighbors
    print(f"Neighbor preservation rate: {metrics['neighbor_preservation']:.4f}")

    # 3. Trustworthiness
    n_neighbors_trust = min(5, len(X_original) - 1)
    metrics['trustworthiness'] = calculate_trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors_trust)
    print(f"Trustworthiness (k={n_neighbors_trust}): {metrics['trustworthiness']:.4f}")

    # 4. Continuity
    metrics['continuity'] = calculate_continuity(X_original, X_embedded, n_neighbors=n_neighbors_trust)
    print(f"Continuity (k={n_neighbors_trust}): {metrics['continuity']:.4f}")

    return metricsimport



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def iterative_label_spilling(X, n_clusters_init=2, max_clusters=15, min_silhouette_improvement=0.01):
    """
    Iterative label spilling clustering method

    Parameters:
    X: Data matrix
    n_clusters_init: Initial number of clusters
    max_clusters: Maximum number of clusters
    min_silhouette_improvement: Minimum silhouette coefficient improvement threshold

    Returns:
    best_labels: Best cluster labels
    best_silhouette: Best silhouette coefficient
    history: Iteration history
    """
    if len(X) < n_clusters_init:
        print(
            f"Warning: Number of data points ({len(X)}) is less than initial number of clusters ({n_clusters_init}), setting initial clusters to 2")
        n_clusters_init = min(2, len(X))

    # Initial clustering
    n_clusters = n_clusters_init
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Ensure enough samples for silhouette calculation (each cluster needs at least 2 samples)
    cluster_sizes = np.bincount(labels)
    if np.any(cluster_sizes < 2):
        print(
            f"Warning: Some clusters have only one sample, cannot calculate silhouette coefficient. Trying fewer clusters.")
        if n_clusters > 2:
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

    try:
        current_silhouette = silhouette_score(X, labels)
        best_labels = labels.copy()
        best_silhouette = current_silhouette
        history = [{'n_clusters': n_clusters, 'silhouette': current_silhouette}]

        print(f"Initial clustering (k={n_clusters}): Silhouette = {current_silhouette:.4f}")

        # Iterate increasing number of clusters
        while n_clusters < max_clusters:
            n_clusters += 1

            # Ensure number of clusters doesn't exceed number of data points
            if n_clusters >= len(X):
                print(
                    f"Number of clusters ({n_clusters}) has reached or exceeded number of data points ({len(X)}), stopping iteration")
                break

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Check for empty clusters
            if len(np.unique(labels)) < n_clusters:
                print(
                    f"Warning: Actual number of clusters ({len(np.unique(labels))}) is less than requested number ({n_clusters}), possible empty clusters")

            # Ensure each cluster has at least 2 samples, otherwise silhouette cannot be calculated
            cluster_sizes = np.bincount(labels)
            if np.any(cluster_sizes < 2):
                print(
                    f"Warning: At k={n_clusters} some clusters have only one sample, cannot calculate silhouette coefficient. Stopping iteration.")
                break

            try:
                new_silhouette = silhouette_score(X, labels)
                history.append({'n_clusters': n_clusters, 'silhouette': new_silhouette})

                print(
                    f"Clustering (k={n_clusters}): Silhouette = {new_silhouette:.4f}, Improvement = {new_silhouette - current_silhouette:.4f}")

                # If silhouette has improved enough, update best result
                if new_silhouette > best_silhouette + min_silhouette_improvement:
                    best_silhouette = new_silhouette
                    best_labels = labels.copy()
                    current_silhouette = new_silhouette
                # If silhouette has decreased significantly, stop iteration
                elif new_silhouette < current_silhouette - min_silhouette_improvement:
                    print(f"Silhouette decreased significantly, stopping iteration")
                    break
                else:
                    current_silhouette = new_silhouette
            except Exception as e:
                print(f"Error calculating silhouette for clustering (k={n_clusters}): {str(e)}")
                break
    except Exception as e:
        print(f"Error in initial clustering (k={n_clusters}): {str(e)}")
        # If initial silhouette cannot be calculated, return initial clustering labels and a very low silhouette
        return labels, -1, [{'n_clusters': n_clusters, 'silhouette': -1}]

    print(f"Best number of clusters: {np.unique(best_labels).size}, Silhouette: {best_silhouette:.4f}")
    return best_labels, best_silhouette, history


def evaluate_embedding(X_original, X_embedded, solubility_values=None, solubility_bins=None):
    """
    Evaluate quality of dimensionality reduction results

    Parameters:
    X_original: Data in original feature space
    X_embedded: Data after dimensionality reduction
    solubility_values: Continuous solubility values (for visualization)
    solubility_bins: Discretized solubility categories (for evaluating clustering quality)

    Returns:
    metrics: Dictionary containing various evaluation metrics
    """
    metrics = {}

    # 1. Silhouette coefficient (using discretized solubility category labels)
    if solubility_bins is not None and len(np.unique(solubility_bins)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_embedded, solubility_bins)
            print(f"Silhouette coefficient using discretized solubility categories: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette coefficient: {str(e)}")
            metrics['silhouette_score'] = None

    # 2. Neighbor preservation rate
    k = min(20, len(X_original) - 1)  # Number of neighbors, not exceeding number of samples minus 1

    # Calculate k nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_original)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_original)

    # Calculate k nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    distances_embedded, indices_embedded = nbrs_embedded.kneighbors(X_embedded)

    # Calculate neighbor preservation rate
    preserved_neighbors = 0
    total_neighbors = 0

    for i in range(len(X_original)):
        # Skip first neighbor (sample itself)
        orig_neighbors = set(indices_orig[i][1:])
        embedded_neighbors = set(indices_embedded[i][1:])
        preserved_neighbors += len(orig_neighbors.intersection(embedded_neighbors))
        total_neighbors += k

    metrics['neighbor_preservation'] = preserved_neighbors / total_neighbors
    print(f"Neighbor preservation rate: {metrics['neighbor_preservation']:.4f}")

    return metrics