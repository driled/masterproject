import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans


def min_toCentroid(df, centroid=None, features=None):
    '''
    Find the closest point in the dataset to a given centroid

    Parameters:
    df: pandas DataFrame with features as columns
    centroid: centroid coordinates (if None, uses mean of all points)
    features: list of feature column names to use

    Returns:
    index of the closest point to the centroid
    '''
    if features is None:
        features = df.columns

    if centroid is None:
        centroid = df[features].mean()

    # Calculate distance from centroid for each point
    dist = df.apply(lambda row: sum(
        [(row[j] - centroid[i]) ** 2 for i, j in enumerate(features)]
    ), axis=1)

    # Return index of minimum distance
    return dist.idxmin()


def ILS(df, labelColumn, outColumn='LS', iterative=True):
    '''
    Apply iterative label spreading in a multi-dimensional feature-space.
    Returns labels for all points and the order-labelled
    and distance-when-labelled for all newly labelled points.

    Parameters:
    df: pandas DataFrame with features as columns and one label column
    labelColumn: Column name that holds initial labels (0 = unlabeled)
    outColumn: Output column name for the new labels
    iterative: If True, apply iteratively; if False, relabel all at once

    Returns:
    newLabels: pandas Series with labels for all points
    orderedLabelled: DataFrame with distances and closest labeled point
    '''
    featureColumns = [i for i in df.columns if i != labelColumn]

    # Keep original index
    oldIndex = df.index
    df = df.reset_index(drop=True)  # Reset to numeric index to avoid problems

    # Separate labelled and unlabelled points
    labelled = df[df[labelColumn] != 0].copy().fillna(0)
    unlabelled = df[df[labelColumn] == 0].copy()

    # Lists for ordered output data
    outD = []  # distances
    outID = []  # unlabelled point IDs
    closeID = []  # closest labelled point IDs

    # Continue while any point is unlabelled
    while len(unlabelled) > 0:
        # Calculate labelled to unlabelled distances matrix (D)
        D = pairwise_distances(
            labelled[featureColumns].values,
            unlabelled[featureColumns].values)

        # Find the minimum distance between a labelled and unlabelled point
        # First the argument in the D matrix
        (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)

        # Then convert to an index ID in the data frame
        idUnL = unlabelled.iloc[posUnL].name
        idL = labelled.iloc[posL].name

        # Switch label from 0 to new label
        unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL, labelColumn]

        # Move newly labelled point to labelled dataframe
        labelled = pd.concat([labelled, unlabelled.loc[[idUnL]]])

        # Drop from unlabelled data frame
        unlabelled = unlabelled.drop(idUnL)

        # Output the distance and ID of the newly labelled point
        outD.append(D.min())
        outID.append(idUnL)
        closeID.append(idL)

    # Throw error if we lost points
    if len(labelled) != len(df):
        raise Exception(
            f"The number of labelled ({len(labelled)}) points "
            f"does not sum to the total ({len(df)})")

    # Create Series for output
    orderLabelled = pd.Series(data=outD, index=outID, name='minR')

    # ID of point label was spread from
    closest = pd.Series(data=closeID, index=outID, name='IDclosestLabel')

    # Rename the label column to output column name
    labelled = labelled.rename(columns={labelColumn: outColumn})

    # Create Series with the new labels
    newLabels = pd.Series(labelled[outColumn].values, index=oldIndex)

    # Return
    return newLabels, pd.concat([orderLabelled, closest], axis=1)


def ILS_clustering(X, n_clusters=4, feature_names=None):
    """
    Apply ILS clustering to a dataset

    Parameters:
    X: numpy array or pandas DataFrame with feature data
    n_clusters: number of clusters to identify
    feature_names: list of feature names (if X is numpy array)

    Returns:
    labels: Cluster labels for each point
    silhouette: Silhouette coefficient
    history: Iteration history (empty for single run)
    """
    # Convert numpy array to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    # Add label column (0 = unlabeled)
    df['label'] = 0

    # Use kmeans to identify initial points to label
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df.drop('label', axis=1))
    centroids = kmeans.cluster_centers_

    # For each cluster, label the point closest to its centroid
    for cluster_id in range(n_clusters):
        # Get points in this cluster
        cluster_points = df.iloc[np.where(kmeans_labels == cluster_id)[0]]

        # Find point closest to centroid
        closest_point_idx = min_toCentroid(
            cluster_points.drop('label', axis=1),
            centroid=centroids[cluster_id]
        )

        # Label the point
        df.loc[closest_point_idx, 'label'] = cluster_id + 1  # +1 so labels start from 1, not 0

    # Run ILS to propagate labels
    new_labels, ordered_info = ILS(df, 'label')

    # Convert labels to zero-based for consistency with other algorithms
    labels = new_labels.values - 1

    # Calculate silhouette score
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1

    # Return labels and empty history (since we're not doing iterative optimization)
    history = [{'n_clusters': n_clusters, 'silhouette': silhouette}]

    return labels, silhouette, history


def ILS_clustering_with_optimization(X, initial_clusters=4, max_clusters=15, min_silhouette_improvement=0.01):
    """
    Apply ILS clustering with iterative optimization for number of clusters

    Parameters:
    X: numpy array or pandas DataFrame with feature data
    initial_clusters: starting number of clusters
    max_clusters: maximum number of clusters to try
    min_silhouette_improvement: minimum improvement in silhouette score to continue

    Returns:
    best_labels: Best cluster labels
    best_silhouette: Best silhouette score
    history: Iteration history
    """
    if len(X) < initial_clusters:
        print(f"Warning: Number of data points ({len(X)}) is less than initial number of clusters "
              f"({initial_clusters}), setting initial clusters to 2")
        initial_clusters = min(2, len(X))

    history = []
    best_labels = None
    best_silhouette = -1
    current_silhouette = -1
    n_clusters = initial_clusters

    print(f"Starting ILS clustering analysis...")

    while n_clusters <= max_clusters:
        try:
            labels, silhouette, _ = ILS_clustering(X, n_clusters=n_clusters)

            history.append({'n_clusters': n_clusters, 'silhouette': silhouette})
            print(f"ILS Clustering (k={n_clusters}): Silhouette = {silhouette:.4f}, "
                  f"Improvement = {silhouette - current_silhouette:.4f}")

            if silhouette > best_silhouette + min_silhouette_improvement:
                best_silhouette = silhouette
                best_labels = labels.copy()
                current_silhouette = silhouette
            else:
                print(f"Stopping iteration: no significant improvement.")
                break

            n_clusters += 1

        except Exception as e:
            print(f"Error in ILS clustering: {str(e)}")
            break

    if best_labels is None:
        # Fallback to basic clustering
        best_labels, best_silhouette, _ = ILS_clustering(X, n_clusters=initial_clusters)
        history = [{'n_clusters': initial_clusters, 'silhouette': best_silhouette}]

    print(f"Best number of clusters: {len(np.unique(best_labels))}, Silhouette: {best_silhouette:.4f}")
    return best_labels, best_silhouette, history


def ILS_clustering_with_solubility(X, solubility_bins):
    """
    Apply ILS clustering using solubility categories as initial labels

    Parameters:
    X: numpy array or pandas DataFrame with feature data
    solubility_bins: array of solubility categories

    Returns:
    labels: Cluster labels
    silhouette: Silhouette coefficient
    history: Iteration history
    """
    # Convert numpy array to DataFrame if needed
    if isinstance(X, np.ndarray):
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    # Number of unique solubility categories
    n_clusters = len(np.unique(solubility_bins))
    print(f"Using {n_clusters} solubility categories as initial labels")

    # Add label column from solubility bins
    df['label'] = solubility_bins + 1  # +1 so 0 can be used for unlabeled points

    # Mask 30% of points to test ILS effectiveness
    n_unlabeled = int(0.3 * len(df))
    mask_indices = np.random.choice(len(df), size=n_unlabeled, replace=False)
    df.loc[mask_indices, 'label'] = 0

    # Run ILS to propagate labels
    new_labels, ordered_info = ILS(df, 'label')

    # Convert labels to zero-based for consistency with other algorithms
    labels = new_labels.values - 1

    # Calculate silhouette score
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1

    history = [{'n_clusters': n_clusters, 'silhouette': silhouette}]

    return labels, silhouette, history