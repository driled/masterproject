import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading


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

    return metrics



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def label_spreading_clustering(X, n_clusters_init=2, max_clusters=15, min_silhouette_improvement=0.01):
    """
    Label Spreading clustering method with iterative optimization

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
        print(f"Warning: Number of data points ({len(X)}) is less than initial number of clusters ({n_clusters_init}), setting initial clusters to 2")
        n_clusters_init = min(2, len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    history = []
    best_labels = None
    best_silhouette = -1
    current_silhouette = -1
    n_clusters = n_clusters_init

    print(f"Starting Label Spreading clustering analysis...")

    while n_clusters < max_clusters:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            initial_labels = kmeans.fit_predict(X_scaled)

            # Mask part of the labels to allow propagation
            masked_labels = initial_labels.copy()
            n_unlabeled = int(0.3 * len(X_scaled))  # 30% masked
            mask_indices = np.random.choice(len(X_scaled), size=n_unlabeled, replace=False)
            masked_labels[mask_indices] = -1

            # Run Label Spreading (drop n_neighbors, not used in rbf kernel)
            label_spreading = LabelSpreading(kernel='rbf', alpha=0.8)
            label_spreading.fit(X_scaled, masked_labels)
            labels = label_spreading.transduction_

            unique_labels = np.unique(labels)
            if len(unique_labels) < n_clusters:
                print(f"Warning: Only {len(unique_labels)} unique clusters found for {n_clusters} requested clusters.")

            cluster_sizes = np.bincount(labels)
            if np.any(cluster_sizes < 2):
                print(f"Warning: Some clusters have only one sample at k={n_clusters}")
                break

            try:
                new_silhouette = silhouette_score(X_scaled, labels)
            except Exception as e:
                print(f"Error calculating silhouette score: {str(e)}")
                break

            history.append({'n_clusters': n_clusters, 'silhouette': new_silhouette})
            print(f"Label Spreading (k={n_clusters}): Silhouette = {new_silhouette:.4f}, Improvement = {new_silhouette - current_silhouette:.4f}")

            if new_silhouette > best_silhouette + min_silhouette_improvement:
                best_silhouette = new_silhouette
                best_labels = labels.copy()
                current_silhouette = new_silhouette
            else:
                print(f"Stopping iteration: no significant improvement.")
                break

            n_clusters += 1

        except Exception as e:
            print(f"Error in Label Spreading clustering: {str(e)}")
            break

    if best_labels is None:
        kmeans = KMeans(n_clusters=n_clusters_init, random_state=42, n_init=10)
        best_labels = kmeans.fit_predict(X_scaled)
        best_silhouette = silhouette_score(X_scaled, best_labels)
        history = [{'n_clusters': n_clusters_init, 'silhouette': best_silhouette}]

    print(f"Best number of clusters: {len(np.unique(best_labels))}, Silhouette: {best_silhouette:.4f}")
    return best_labels, best_silhouette, history


def label_spreading_clustering_with_solubility(X, solubility_bins, max_clusters=15, min_silhouette_improvement=0.01):
    """
    使用溶解度分类作为初始标签的标签传播聚类方法

    参数:
    X: 降维后的数据矩阵
    solubility_bins: 基于溶解度的类别标签
    max_clusters: 最大簇数量
    min_silhouette_improvement: 最小轮廓系数改进阈值

    返回:
    best_labels: 最佳聚类标签
    best_silhouette: 最佳轮廓系数
    history: 迭代历史
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    history = []

    # 使用溶解度类别作为初始标签
    n_clusters = len(np.unique(solubility_bins))
    initial_labels = solubility_bins.copy()

    # 掩盖部分标签以允许传播
    masked_labels = initial_labels.copy()
    n_unlabeled = int(0.3 * len(X_scaled))  # 30% 被掩盖
    mask_indices = np.random.choice(len(X_scaled), size=n_unlabeled, replace=False)
    masked_labels[mask_indices] = -1

    # 运行标签传播
    label_spreading = LabelSpreading(kernel='rbf', alpha=0.8)
    label_spreading.fit(X_scaled, masked_labels)
    labels = label_spreading.transduction_

    # 计算轮廓系数
    try:
        silhouette = silhouette_score(X_scaled, labels)
        history.append({'n_clusters': n_clusters, 'silhouette': silhouette})
        print(f"基于溶解度的标签传播 (k={n_clusters}): 轮廓系数 = {silhouette:.4f}")
    except Exception as e:
        print(f"计算轮廓系数时出错: {str(e)}")
        silhouette = -1

    return labels, silhouette, history

