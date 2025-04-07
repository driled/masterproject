"""
Divergence metrics for evaluating dimensionality reduction methods
"""

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras import backend as K


def jensen_shannon_divergence(P, Q):
    """
    Calculate Jensen-Shannon divergence between two distributions P and Q

    Parameters:
    P: First probability distribution
    Q: Second probability distribution

    Returns:
    js_divergence: Jensen-Shannon divergence value
    """
    # Ensure proper normalization
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Calculate the average distribution
    M = (P + Q) / 2

    # Calculate Jensen-Shannon Divergence
    js_divergence = 0.5 * entropy(P, M) + 0.5 * entropy(Q, M)

    return js_divergence


def calculate_pca_divergence(X_original, X_pca, pca_model):
    """
    Calculate divergence for PCA using Jensen-Shannon divergence between
    original data and reconstructed data distributions

    Parameters:
    X_original: Original data matrix
    X_pca: Reduced dimensionality data
    pca_model: Fitted PCA model

    Returns:
    divergence: Divergence value
    """
    # Reconstruct data from PCA
    X_reconstructed = pca_model.inverse_transform(X_pca)

    # Calculate distance matrices (distribution of pairwise distances)
    dist_original = pdist(X_original, 'euclidean')
    dist_reconstructed = pdist(X_reconstructed, 'euclidean')

    # Normalize to create probability distributions
    hist_original, _ = np.histogram(dist_original, bins=50, density=True)
    hist_reconstructed, _ = np.histogram(dist_reconstructed, bins=50, density=True)

    # Add small constant to avoid zeros
    hist_original = hist_original + 1e-10
    hist_reconstructed = hist_reconstructed + 1e-10

    # Calculate Jensen-Shannon divergence
    js_div = jensen_shannon_divergence(hist_original, hist_reconstructed)

    return js_div


def calculate_tsne_divergence(X_original, X_tsne, perplexity=30):
    """
    Estimate KL divergence for t-SNE
    Note: t-SNE already optimizes KL divergence, but we can estimate it

    Parameters:
    X_original: Original data matrix
    X_tsne: Reduced dimensionality data
    perplexity: Perplexity parameter used in t-SNE

    Returns:
    divergence: Estimated divergence value
    """
    # Calculate high-dimensional pairwise similarities (P)
    n_samples = X_original.shape[0]

    # Compute pairwise distances
    dist_matrix = squareform(pdist(X_original, 'euclidean'))

    # Calculate P (high-dimensional similarities)
    # This is a simplified version - t-SNE implements a more complex version
    sigma = np.sqrt(perplexity / 2)
    P = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(P, 0)
    P = P / np.sum(P)

    # Calculate Q (low-dimensional similarities)
    dist_matrix_low = squareform(pdist(X_tsne, 'euclidean'))
    Q = 1 / (1 + dist_matrix_low ** 2)
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q)

    # Calculate KL divergence between P and Q
    P = P + 1e-10  # Add small constant to avoid log(0)
    Q = Q + 1e-10

    kl_div = np.sum(P * np.log(P / Q))

    return kl_div


def calculate_umap_divergence(X_original, X_umap, n_neighbors=15):
    """
    Estimate UMAP's cross-entropy between high and low dimensional representations

    Parameters:
    X_original: Original data matrix
    X_umap: Reduced dimensionality data
    n_neighbors: Number of neighbors parameter used in UMAP

    Returns:
    divergence: Estimated divergence value
    """
    # Calculate high-dimensional nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_original)
    high_dist, high_indices = nbrs.kneighbors(X_original)

    # Calculate low-dimensional nearest neighbors
    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_umap)
    low_dist, low_indices = nbrs_low.kneighbors(X_umap)

    # Calculate cross-entropy based on neighborhood preservation
    cross_entropy = 0
    for i in range(len(X_original)):
        # High-dimensional neighbors
        high_neighbors = set(high_indices[i])
        # Low-dimensional neighbors
        low_neighbors = set(low_indices[i])
        # Intersection (preserved neighbors)
        preserved = len(high_neighbors.intersection(low_neighbors))
        # Calculate probability
        prob_preserved = preserved / n_neighbors
        # Add to cross entropy (avoid log(0))
        if prob_preserved > 0:
            cross_entropy -= np.log(prob_preserved)

    # Normalize by number of points
    cross_entropy /= len(X_original)

    return cross_entropy


def calculate_autoencoder_divergence(X_original, encoder, decoder):
    """
    Calculate divergence for Autoencoder using Jensen-Shannon divergence

    Parameters:
    X_original: Original data matrix
    encoder: Encoder model
    decoder: Decoder model

    Returns:
    divergence: Divergence value
    """
    # Get encoded representation
    X_encoded = encoder.predict(X_original)

    # Reconstruct from encoded representation
    X_reconstructed = decoder.predict(X_encoded)

    # Calculate distance matrices
    dist_original = pdist(X_original, 'euclidean')
    dist_reconstructed = pdist(X_reconstructed, 'euclidean')

    # Normalize to create probability distributions
    hist_original, _ = np.histogram(dist_original, bins=50, density=True)
    hist_reconstructed, _ = np.histogram(dist_reconstructed, bins=50, density=True)

    # Add small constant to avoid zeros
    hist_original = hist_original + 1e-10
    hist_reconstructed = hist_reconstructed + 1e-10

    # Calculate Jensen-Shannon divergence
    js_div = jensen_shannon_divergence(hist_original, hist_reconstructed)

    return js_div


def calculate_vae_kl_divergence(z_mean, z_log_var):
    """
    Calculate the KL divergence for a VAE model
    This is a TensorFlow implementation

    Parameters:
    z_mean: Mean values of the latent space
    z_log_var: Log variance values of the latent space

    Returns:
    kl_loss: KL divergence loss (scalar value)
    """
    # KL divergence between the learned distribution and a standard normal distribution
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))

    # Convert tensor to Python float if needed
    if hasattr(kl_loss, 'numpy'):
        kl_loss = float(kl_loss.numpy())

    return kl_loss


def numpy_calculate_vae_kl_divergence(z_mean, z_log_var):
    """
    Calculate the KL divergence for a VAE model
    This is a NumPy implementation for when TensorFlow is not available

    Parameters:
    z_mean: Mean values of the latent space
    z_log_var: Log variance values of the latent space

    Returns:
    kl_loss: KL divergence loss (scalar value)
    """
    # KL divergence between the learned distribution and a standard normal distribution
    kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))

    return float(kl_loss)


def calculate_manifold_divergence(X_original, X_embedded, n_neighbors=15):
    """
    Calculate manifold divergence based on geodesic distance preservation

    Parameters:
    X_original: Original data matrix
    X_embedded: Embedded data after dimensionality reduction
    n_neighbors: Number of neighbors to consider for geodesic distances

    Returns:
    divergence: Manifold divergence score (higher means more distortion)
    """
    from sklearn.manifold import SpectralEmbedding

    # Compute approximate geodesic distances in original space
    # (Using Spectral Embedding's internal representation)
    spectral = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors,
                                 eigen_solver='arpack', random_state=42)

    # We don't need the embedding, just using this to compute the affinity matrix
    try:
        spectral.fit(X_original)
        dist_original = spectral.affinity_matrix_.toarray()

        # Compute Euclidean distances in embedding space
        dist_embedded = squareform(pdist(X_embedded, 'euclidean'))

        # Normalize both distance matrices
        dist_original = dist_original / np.max(dist_original)
        dist_embedded = dist_embedded / np.max(dist_embedded)

        # Calculate the divergence as the mean squared difference
        # between normalized distance matrices
        divergence = np.mean((dist_original - dist_embedded) ** 2)

        return divergence
    except:
        # Fallback if spectral embedding fails
        return None