import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.impute import SimpleImputer


def handle_nan_values(X):
    """
    Check for and handle NaN values in the data

    Parameters:
    X: Input data matrix

    Returns:
    X_clean: Data matrix with NaN values handled
    has_nan: Boolean indicating if the original data had NaN values
    """
    has_nan = np.isnan(X).any()

    if has_nan:
        print("Warning: NaN values detected in the data. Applying imputation...")
        imputer = SimpleImputer(strategy='mean')
        X_clean = imputer.fit_transform(X)
        print(f"Imputation completed. Shape preserved: {X.shape} -> {X_clean.shape}")
        return X_clean, True

    return X, False


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
    # Handle NaN values if present
    X_original, _ = handle_nan_values(X_original)
    X_embedded, _ = handle_nan_values(X_embedded)

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
    # Handle NaN values if present
    X_original, _ = handle_nan_values(X_original)
    X_embedded, _ = handle_nan_values(X_embedded)

    # Calculate k-nearest neighbors in original space
    nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_original)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_original)

    # Calculate k-nearest neighbors in embedded space
    nbrs_embed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded)
    distances_embedded, indices_embed = nbrs_embed.kneighbors(X_embedded)

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


def evaluate_embedding(X_original, X_embedded, energy_values=None, energy_bins=None):
    """
    Evaluate quality of dimensionality reduction results with multiple neighborhood sizes

    Parameters:
    X_original: Data in original feature space
    X_embedded: Data after dimensionality reduction
    energy_values: Continuous energy values (for visualization)
    energy_bins: Discretized energy categories (for evaluating clustering quality)

    Returns:
    metrics: Dictionary containing various evaluation metrics
    """
    # Handle NaN values if present
    X_original, _ = handle_nan_values(X_original)
    X_embedded, had_nan = handle_nan_values(X_embedded)

    metrics = {}

    # If we had to impute values, record this in metrics
    if had_nan:
        metrics['had_nan_values'] = True
        print("Warning: NaN values were detected and imputed in the embedded data")

    # 1. Silhouette coefficient (using discretized energy category labels)
    if energy_bins is not None and len(np.unique(energy_bins)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_embedded, energy_bins)
            print(f"Silhouette coefficient using discretized energy categories: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette coefficient: {str(e)}")
            metrics['silhouette_score'] = None

    # 2. Neighbor preservation rate (with original k=20 parameter)
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

    # 3. Trustworthiness and Continuity with multiple neighborhood sizes
    # Define a range of neighborhood sizes to test
    neighbor_sizes = [5, 10, 20, min(50, len(X_original) // 4)]

    print("\nEvaluating metrics across different neighborhood sizes:")
    for n_size in neighbor_sizes:
        # Skip if neighborhood size is too large
        if n_size >= len(X_original):
            print(f"  Skipping k={n_size} (exceeds data size)")
            continue

        # Calculate metrics for this neighborhood size
        key_suffix = f"_k{n_size}"

        # Calculate trustworthiness
        trust_value = calculate_trustworthiness(X_original, X_embedded, n_neighbors=n_size)
        metrics[f'trustworthiness{key_suffix}'] = trust_value

        # Calculate continuity
        cont_value = calculate_continuity(X_original, X_embedded, n_neighbors=n_size)
        metrics[f'continuity{key_suffix}'] = cont_value

        print(f"  Neighborhood size k={n_size}:")
        print(f"    Trustworthiness: {trust_value:.4f}")
        print(f"    Continuity: {cont_value:.4f}")

    # Keep the default k=5 metrics for backward compatibility
    default_k = min(5, len(X_original) - 1)
    metrics['trustworthiness'] = metrics.get(f'trustworthiness_k{default_k}',
                                             calculate_trustworthiness(X_original, X_embedded, n_neighbors=default_k))
    metrics['continuity'] = metrics.get(f'continuity_k{default_k}',
                                        calculate_continuity(X_original, X_embedded, n_neighbors=default_k))

    return metrics


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler


import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.semi_supervised import LabelSpreading

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



# 更新结果汇总函数，将多个邻域大小的结果包含在内
def prepare_results_summary(all_results):
    """
    Prepare comprehensive results summary including metrics with different neighborhood sizes

    Parameters:
    all_results: List of results dictionaries from all dimensionality reduction methods

    Returns:
    results_df: DataFrame with comprehensive results summary
    """
    # 基本结果数据
    base_metrics = {
        'Method': [r['method'] for r in all_results],
        'Parameters': [r['params'] for r in all_results],
        'Runtime (s)': [r['runtime'] for r in all_results],
        'Neighbor Preservation': [r['metrics'].get('neighbor_preservation', None) for r in all_results],
        'Silhouette Score': [r['metrics'].get('silhouette_score', None) for r in all_results],
        # 默认参数的trustworthiness和continuity，保持向后兼容
        'Trustworthiness': [r['metrics'].get('trustworthiness', None) for r in all_results],
        'Continuity': [r['metrics'].get('continuity', None) for r in all_results],
        'Variance Explained': [r['metrics'].get('variance_explained', None) for r in all_results],
        'Reconstruction Error': [r['metrics'].get('reconstruction_error', None) for r in all_results],
        'KL Divergence': [r['metrics'].get('kl_divergence', None) for r in all_results],
    }

    # 创建基本DataFrame
    results_df = pd.DataFrame(base_metrics)

    # 添加不同邻域大小的指标
    neighborhood_sizes = [5, 10, 20, 50]
    for k in neighborhood_sizes:
        # 检查是否有任何结果包含此邻域大小的指标
        trust_key = f'trustworthiness_k{k}'
        cont_key = f'continuity_k{k}'

        has_metrics = any(trust_key in r['metrics'] or cont_key in r['metrics'] for r in all_results)

        if has_metrics:
            # 添加此邻域大小的列
            results_df[f'Trustworthiness (k={k})'] = [r['metrics'].get(trust_key, None) for r in all_results]
            results_df[f'Continuity (k={k})'] = [r['metrics'].get(cont_key, None) for r in all_results]

    return results_df