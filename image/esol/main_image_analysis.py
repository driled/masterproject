"""
Main module for ESOL image-based dimensionality reduction and clustering analysis with ILS
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from torch._functorch._aot_autograd.logging_utils import model_name

from image_data_loader import load_image_data
from cnn_feature_extractor import extract_features_from_images
from pub_func.table.ils_clustering import ILS_clustering, ILS_clustering_with_optimization, ILS_clustering_with_solubility
from pub_func.table.dim_reduction import perform_pca, perform_umap, perform_autoencoder
from pub_func.table.clustering import evaluate_embedding
from pub_func.table.visualization import (save_embedding_csv, plot_2d_embedding,
                           plot_clustering_result, plot_silhouette_history,
                           plot_comparison_bar, save_results_summary)
from pub_func.table.divergence_metrics import (calculate_pca_divergence, calculate_umap_divergence,
                                calculate_manifold_divergence)


def run_image_dimensionality_reduction(image_dir, metadata_path, output_path, model_name='resnet18'):
    """
    Main function for running image-based dimensionality reduction, clustering, and evaluation
    with Iterative Label Spreading (ILS) clustering for ESOL dataset

    Parameters:
    image_dir: Directory containing molecule images
    metadata_path: Path to metadata CSV file with solubility values
    output_path: Output results path
    model_name: CNN model to use for feature extraction

    Returns:
    results_df: DataFrame with dimensionality reduction results
    clustering_df: DataFrame with clustering results
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load image data
    image_files, solubility_values, solubility_bins, molecule_ids = load_image_data(image_dir, metadata_path)

    # Extract CNN features
    print(f"\nExtracting CNN features using {model_name}...")
    start_time = time.time()
    features, _ = extract_features_from_images(image_files, solubility_values, model_name=model_name)
    feature_extraction_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")

    # Save extracted features
    feature_df = pd.DataFrame(features)
    if molecule_ids:
        feature_df['molecule_id'] = molecule_ids
    feature_df.to_csv(os.path.join(output_path, f"{model_name}_features.csv"), index=False)
    print(f"Extracted features saved to {os.path.join(output_path, f'{model_name}_features.csv')}")

    # Perform clustering on original high-dimensional features
    print("\nPerforming ILS clustering on original high-dimensional CNN features...")
    cluster_start_time = time.time()

    # Use either solubility-based or optimization-based clustering
    if solubility_bins is not None:
        print(f"Applying solubility-based ILS clustering...")
        original_cluster_labels, original_silhouette, original_history = ILS_clustering_with_solubility(features,
                                                                                                        solubility_bins)
    else:
        # Use iterative optimization
        original_cluster_labels, original_silhouette, original_history = ILS_clustering_with_optimization(features)

    cluster_runtime = time.time() - cluster_start_time
    print(f"High-dimensional clustering completed in {cluster_runtime:.2f} seconds")
    print(f"Number of clusters: {len(np.unique(original_cluster_labels))}, Silhouette Score: {original_silhouette:.4f}")

    # Save original clustering results
    original_clustering_df = pd.DataFrame({
        'Method': 'Original CNN Features',
        'Parameters': f'model={model_name}, dim={features.shape[1]}',
        'n_clusters': len(np.unique(original_cluster_labels)),
        'silhouette_score': original_silhouette,
        'runtime': cluster_runtime
    }, index=[0])
    save_results_summary(original_clustering_df, output_path, 'original_clustering_results.csv')

    # Initialize results containers
    all_results = []
    all_clustering_results = []

    # Run PCA analysis for visualization (using original clusters)
    pca_results = run_pca_analysis(features, original_cluster_labels, solubility_values, solubility_bins, output_path)
    all_results.extend(pca_results)
    all_clustering_results.append({
        'method': 'PCA + Original Clustering',
        'params': f'High-dim CNN features clustering',
        'n_clusters': len(np.unique(original_cluster_labels)),
        'silhouette_score': original_silhouette,
        'cluster_history': original_history
    })

    # Run UMAP analysis for visualization (using original clusters)
    umap_results = run_umap_analysis(features, original_cluster_labels, solubility_values, solubility_bins, output_path)
    all_results.extend(umap_results)
    all_clustering_results.append({
        'method': 'UMAP + Original Clustering',
        'params': f'High-dim CNN features clustering',
        'n_clusters': len(np.unique(original_cluster_labels)),
        'silhouette_score': original_silhouette,
        'cluster_history': original_history
    })

    # Run Autoencoder analysis for visualization (using original clusters)
    ae_results = run_autoencoder_analysis(features, original_cluster_labels, solubility_values, solubility_bins,
                                          output_path)
    all_results.extend(ae_results)
    all_clustering_results.append({
        'method': 'Autoencoder + Original Clustering',
        'params': f'High-dim CNN features clustering',
        'n_clusters': len(np.unique(original_cluster_labels)),
        'silhouette_score': original_silhouette,
        'cluster_history': original_history
    })

    # Prepare comprehensive results summary
    results_df = prepare_results_summary(all_results)

    # Add CNN model information to results
    results_df['CNN Model'] = model_name
    results_df['Feature Extraction Time (s)'] = feature_extraction_time

    # Prepare clustering results summary
    clustering_df = pd.DataFrame({
        'Method': [r['method'] for r in all_clustering_results],
        'Parameters': [r['params'] for r in all_clustering_results],
        'Optimal Clusters': [r['n_clusters'] for r in all_clustering_results],
        'Silhouette Score': [r['silhouette_score'] for r in all_clustering_results],
        'CNN Model': model_name
    })

    # Save results summaries
    save_results_summary(results_df, output_path, 'dimensionality_reduction_results.csv')
    save_results_summary(clustering_df, output_path, 'clustering_results.csv')

    # Evaluate with true labels (solubility categories)
    if solubility_bins is not None:
        true_labels_df = evaluate_with_true_labels(features, solubility_bins, output_path)

    # Create comprehensive comparison visualizations
    create_comprehensive_visualizations(results_df, clustering_df, output_path)

    # Generate metrics correlation analysis
    correlation_analysis(results_df, output_path)

    print(f"\nAll analyses completed! Results saved to {output_path}")
    return results_df, clustering_df


def run_pca_analysis(features, original_cluster_labels, solubility_values, solubility_bins, output_path):
    """
    Run PCA dimensionality reduction for visualization
    Uses clustering results from original high-dimensional features

    Parameters:
    features: CNN feature data
    original_cluster_labels: Cluster labels from high-dimensional clustering
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    """
    results = []

    print("\nRunning PCA dimensionality reduction on CNN features...")
    for n_components in [2, 5, 10, 20]:
        print(f"  n_components = {n_components}")
        start_time = time.time()

        # Perform PCA
        X_pca, pca_model, pca_metrics = perform_pca(features, n_components)
        runtime = time.time() - start_time

        # Calculate divergence metrics
        print("  Calculating divergence metrics...")
        try:
            pca_divergence = calculate_pca_divergence(features, X_pca, pca_model)
            manifold_divergence = calculate_manifold_divergence(features, X_pca)

            # Add divergence metrics to pca_metrics
            pca_metrics['pca_divergence'] = pca_divergence
            pca_metrics['manifold_divergence'] = manifold_divergence

            print(f"  PCA Divergence: {pca_divergence:.4f}")
            print(
                f"  Manifold Divergence: {manifold_divergence:.4f}" if manifold_divergence is not None else "  Manifold Divergence: N/A")
        except Exception as e:
            print(f"  Error calculating divergence metrics: {str(e)}")

        # Save reduction results
        save_embedding_csv(X_pca, output_path, f"pca_{n_components}d.csv")

        # Evaluate reduction quality
        eval_metrics = evaluate_embedding(features, X_pca, solubility_values, solubility_bins)

        # Merge all metrics
        all_metrics = {**eval_metrics, **pca_metrics}

        # Record results
        results.append({
            'method': 'PCA',
            'params': f'n_components={n_components}',
            'runtime': runtime,
            'metrics': all_metrics
        })

        # If 2D, create visualizations
        if n_components == 2:
            # Continuous value coloring
            plot_2d_embedding(
                X_pca, solubility_values, output_path, 'pca_2d_plot.png',
                'PCA 2D Projection of CNN Features', 'viridis', 'Solubility Value'
            )

            # Discrete category coloring
            if solubility_bins is not None:
                plot_2d_embedding(
                    X_pca, solubility_bins, output_path, 'pca_2d_plot_discrete.png',
                    'PCA 2D Projection of CNN Features (Colored by Solubility Category)', 'tab10', 'Solubility Category'
                )

            # Visualization with high-dimensional clustering results
            plot_clustering_result(
                X_pca, original_cluster_labels, output_path, f'pca_{n_components}d_high_dim_clusters.png',
                f'PCA 2D Projection - Clusters from High-Dimensional CNN Features ({len(np.unique(original_cluster_labels))} clusters)'
            )

    return results


def run_umap_analysis(features, original_cluster_labels, solubility_values, solubility_bins, output_path):
    """
    Run UMAP dimensionality reduction for visualization
    Uses clustering results from original high-dimensional features

    Parameters:
    features: CNN feature data
    original_cluster_labels: Cluster labels from high-dimensional clustering
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    """
    results = []

    print("\nRunning UMAP dimensionality reduction on CNN features...")
    for n_neighbors in [5, 15, 30]:
        for min_dist in [0.1, 0.5]:
            for n_components in [2, 5]:
                print(f"  n_neighbors = {n_neighbors}, min_dist = {min_dist}, n_components = {n_components}")
                start_time = time.time()

                # Perform UMAP
                X_umap, umap_model, umap_metrics = perform_umap(features, n_components, n_neighbors, min_dist)
                runtime = time.time() - start_time

                # Calculate divergence metrics
                print("  Calculating divergence metrics...")
                try:
                    umap_divergence = calculate_umap_divergence(features, X_umap, n_neighbors)
                    manifold_divergence = calculate_manifold_divergence(features, X_umap)

                    # Add divergence metrics to umap_metrics
                    umap_metrics['umap_divergence'] = umap_divergence
                    umap_metrics['manifold_divergence'] = manifold_divergence

                    print(f"  UMAP Divergence: {umap_divergence:.4f}")
                    print(
                        f"  Manifold Divergence: {manifold_divergence:.4f}" if manifold_divergence is not None else "  Manifold Divergence: N/A")
                except Exception as e:
                    print(f"  Error calculating divergence metrics: {str(e)}")

                # Save reduction results
                save_embedding_csv(X_umap, output_path, f"umap_nn{n_neighbors}_md{min_dist}_{n_components}d.csv")

                # Evaluate reduction quality
                eval_metrics = evaluate_embedding(features, X_umap, solubility_values, solubility_bins)

                # Merge all metrics
                all_metrics = {**eval_metrics, **umap_metrics}

                # Record results
                results.append({
                    'method': 'UMAP',
                    'params': f'n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}',
                    'runtime': runtime,
                    'metrics': all_metrics
                })

                # If 2D, create visualizations
                if n_components == 2:
                    # Continuous value coloring
                    plot_2d_embedding(
                        X_umap, solubility_values, output_path, f'umap_nn{n_neighbors}_md{min_dist}_2d_plot.png',
                        f'UMAP 2D Projection of CNN Features (n_neighbors={n_neighbors}, min_dist={min_dist})',
                        'viridis',
                        'Solubility Value'
                    )

                    # Discrete category coloring
                    if solubility_bins is not None:
                        plot_2d_embedding(
                            X_umap, solubility_bins, output_path,
                            f'umap_nn{n_neighbors}_md{min_dist}_2d_plot_discrete.png',
                            f'UMAP 2D Projection of CNN Features (n_neighbors={n_neighbors}, min_dist={min_dist}, Colored by Solubility Category)',
                            'tab10', 'Solubility Category'
                        )

                    # Visualization with high-dimensional clustering results
                    plot_clustering_result(
                        X_umap, original_cluster_labels, output_path,
                        f'umap_nn{n_neighbors}_md{min_dist}_{n_components}d_high_dim_clusters.png',
                        f'UMAP 2D Projection - Clusters from High-Dimensional CNN Features ({len(np.unique(original_cluster_labels))} clusters)'
                    )

    return results


def run_autoencoder_analysis(features, original_cluster_labels, solubility_values, solubility_bins, output_path):
    """
    Run autoencoder dimensionality reduction for visualization
    Uses clustering results from original high-dimensional features

    Parameters:
    features: CNN feature data
    original_cluster_labels: Cluster labels from high-dimensional clustering
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    """
    results = []

    print("\nRunning autoencoder dimensionality reduction on CNN features...")

    # Try different encoding dimensions and intermediate layer sizes
    for encoding_dim in [2, 5, 10, 20]:
        for intermediate_dim in [128, 256, 512]:
            print(f"  encoding_dim = {encoding_dim}, intermediate_dim = {intermediate_dim}")
            start_time = time.time()

            # Perform autoencoder dimensionality reduction
            X_ae, encoder, ae_metrics = perform_autoencoder(features, encoding_dim, intermediate_dim)
            runtime = time.time() - start_time

            # Calculate divergence metrics
            print("  Calculating divergence metrics...")
            try:
                # The decoder is needed for autoencoder divergence
                # For simplicity, we'll use manifold divergence which doesn't need the decoder
                manifold_divergence = calculate_manifold_divergence(features, X_ae)

                # Add divergence metrics to ae_metrics
                ae_metrics['manifold_divergence'] = manifold_divergence

                print(
                    f"  Manifold Divergence: {manifold_divergence:.4f}" if manifold_divergence is not None else "  Manifold Divergence: N/A")
            except Exception as e:
                print(f"  Error calculating divergence metrics: {str(e)}")

            # Save reduction results
            save_embedding_csv(X_ae, output_path, f"ae_ed{encoding_dim}_id{intermediate_dim}.csv")

            # Evaluate reduction quality
            eval_metrics = evaluate_embedding(features, X_ae, solubility_values, solubility_bins)

            # Merge all metrics
            all_metrics = {**eval_metrics, **ae_metrics}

            # Record results
            results.append({
                'method': 'Autoencoder',
                'params': f'encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}',
                'runtime': runtime,
                'metrics': all_metrics
            })

            # If 2D, create visualizations
            if encoding_dim == 2:
                # Continuous value coloring
                plot_2d_embedding(
                    X_ae, solubility_values, output_path, f'ae_ed{encoding_dim}_id{intermediate_dim}_2d_plot.png',
                    f'AE 2D Projection of CNN Features (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim})',
                    'viridis', 'Solubility Value'
                )

                # Discrete category coloring
                if solubility_bins is not None:
                    plot_2d_embedding(
                        X_ae, solubility_bins, output_path,
                        f'ae_ed{encoding_dim}_id{intermediate_dim}_2d_plot_discrete.png',
                        f'AE 2D Projection of CNN Features (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}, Colored by Solubility Category)',
                        'tab10', 'Solubility Category'
                    )

                # Visualization with high-dimensional clustering results
                plot_clustering_result(
                    X_ae, original_cluster_labels, output_path,
                    f'ae_ed{encoding_dim}_id{intermediate_dim}_high_dim_clusters.png',
                    f'AE 2D Projection - Clusters from High-Dimensional CNN Features ({len(np.unique(original_cluster_labels))} clusters)'
                )

    return results


def evaluate_with_true_labels(features, solubility_bins, output_path):
    """
    Calculate silhouette scores using true solubility categories

    Parameters:
    features: CNN feature data
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    true_labels_df: DataFrame with silhouette scores for true labels
    """
    if solubility_bins is None:
        return None

    silhouette_with_true_labels = []

    # Calculate silhouette score in original feature space
    try:
        orig_s_score = silhouette_score(features, solubility_bins)
        silhouette_with_true_labels.append({
            'Method': 'Original CNN Features',
            'Parameters': f'dim={features.shape[1]}',
            'Silhouette Score (True Labels)': orig_s_score
        })
        print(f"Silhouette score using true labels in original feature space: {orig_s_score:.4f}")
    except:
        print("Could not calculate silhouette score in original feature space")

    # PCA dimensionality reduction
    for n_components in [2, 5, 10, 20]:
        X_reduced = perform_pca(features, n_components)[0]
        try:
            s_score = silhouette_score(X_reduced, solubility_bins)
            silhouette_with_true_labels.append({
                'Method': 'PCA',
                'Parameters': f'n_components={n_components}',
                'Silhouette Score (True Labels)': s_score
            })
        except:
            pass

    # UMAP dimensionality reduction
    for n_neighbors in [5, 15, 30]:
        for min_dist in [0.1, 0.5]:
            for n_components in [2, 5]:
                X_reduced = perform_umap(features, n_components, n_neighbors, min_dist)[0]
                try:
                    s_score = silhouette_score(X_reduced, solubility_bins)
                    silhouette_with_true_labels.append({
                        'Method': 'UMAP',
                        'Parameters': f'n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}',
                        'Silhouette Score (True Labels)': s_score
                    })
                except:
                    pass

    # Create DataFrame
    true_labels_df = pd.DataFrame(silhouette_with_true_labels)

    # Save results
    save_results_summary(true_labels_df, output_path, 'silhouette_with_true_labels.csv')

    return true_labels_df


def correlation_analysis(results_df, output_path):
    """
    Analyze correlations between different evaluation metrics

    Parameters:
    results_df: DataFrame with dimensionality reduction results
    output_path: Output directory path
    """
    # Select numerical metrics columns
    metric_columns = [
        'Neighbor Preservation', 'Silhouette Score', 'Trustworthiness',
        'Continuity', 'Runtime (s)', 'Reconstruction Error',
        'PCA Divergence', 'UMAP Divergence', 'Manifold Divergence'
    ]

    # Filter to include only columns that exist in the dataframe
    available_metrics = [col for col in metric_columns if col in results_df.columns]

    # Filter rows with non-null values for the relevant metrics
    metrics_data = results_df[available_metrics].dropna(how='all')

    if len(metrics_data) > 1:  # Ensure there's enough data for correlation
        try:
            # Calculate correlation matrix
            corr_matrix = metrics_data.corr()

            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.colorbar(label='Correlation Coefficient')
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            plt.title('Correlation Between Evaluation Metrics')

            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                             ha='center', va='center',
                             color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'metrics_correlation.png'))
            plt.close()

            # Save correlation matrix to CSV
            corr_matrix.to_csv(os.path.join(output_path, 'metrics_correlation.csv'))
            print(f"Metrics correlation analysis saved to {output_path}")
        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")


def prepare_results_summary(all_results):
    """
    Prepare comprehensive results summary from all dimensionality reduction methods

    Parameters:
    all_results: List of results dictionaries from all dimensionality reduction methods

    Returns:
    results_df: DataFrame with comprehensive results summary
    """
    results_df = pd.DataFrame({
        'Method': [r['method'] for r in all_results],
        'Parameters': [r['params'] for r in all_results],
        'Runtime (s)': [r['runtime'] for r in all_results],
        'Neighbor Preservation': [r['metrics'].get('neighbor_preservation', None) for r in all_results],
        'Silhouette Score': [r['metrics'].get('silhouette_score', None) for r in all_results],
        'Trustworthiness': [r['metrics'].get('trustworthiness', None) for r in all_results],
        'Continuity': [r['metrics'].get('continuity', None) for r in all_results],
        'Variance Explained': [r['metrics'].get('variance_explained', None) for r in all_results],
        'Reconstruction Error': [r['metrics'].get('reconstruction_error', None) for r in all_results],
        'PCA Divergence': [r['metrics'].get('pca_divergence', None) for r in all_results],
        'UMAP Divergence': [r['metrics'].get('umap_divergence', None) for r in all_results],
        'Manifold Divergence': [r['metrics'].get('manifold_divergence', None) for r in all_results],
    })

    return results_df


def create_comprehensive_visualizations(results_df, clustering_df, output_path):
    """
    Create comprehensive comparative visualizations of dimensionality reduction and clustering results

    Parameters:
    results_df: DataFrame with dimensionality reduction results
    clustering_df: DataFrame with clustering results
    output_path: Output directory path
    """
    metrics_to_plot = [
        ('Runtime (s)', 'Runtime Comparison'),
        ('Neighbor Preservation', 'Neighbor Preservation Comparison'),
        ('Trustworthiness', 'Trustworthiness Comparison'),
        ('Continuity', 'Continuity Comparison'),
        ('Silhouette Score', 'Silhouette Score Comparison'),
    ]

    # Plot each metric
    for metric, title in metrics_to_plot:
        if metric in results_df.columns and any(~results_df[metric].isna()):
            plot_comparison_bar(
                results_df, 'Dimensionality Reduction Method', metric,
                f'{title} of Dimensionality Reduction Methods', output_path,
                f'{metric.lower().replace(" ", "_")}_comparison.png'
            )

    # Plot divergence metrics
    divergence_metrics = [
        ('PCA Divergence', 'PCA Divergence Comparison'),
        ('UMAP Divergence', 'UMAP Divergence Comparison'),
        ('Manifold Divergence', 'Manifold Divergence Comparison')
    ]

    for metric, title in divergence_metrics:
        if metric in results_df.columns and any(~results_df[metric].isna()):
            plot_comparison_bar(
                results_df, 'Dimensionality Reduction Method', metric,
                f'{title} of Dimensionality Reduction Methods', output_path,
                f'{metric.lower().replace(" ", "_")}_comparison.png'
            )

    # Clustering silhouette score comparison
    plot_comparison_bar(
        clustering_df, 'Method', 'Silhouette Score',
        'Clustering Silhouette Score Comparison', output_path,
        'clustering_silhouette_comparison.png'
    )


if __name__ == "__main__":
    # Set input and output paths
    image_dir = r"D:\materproject\all-reps\ESOL\ESOL-image"
    metadata_path = r"D:\materproject\all-reps\ESOL\ESOL-table\esol.csv"  # Path to CSV with solubility data
    output_path = r"D:\materproject\single-rep-rd\image\ESOL"

    # Run main function with ResNet18 for feature extraction
    results_df, clustering_df = run_image_dimensionality_reduction(image_dir, metadata_path, output_path,
                                                                   model_name='resnet18')

    # Print results summary
    print("\nDimensionality reduction results summary:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df)

    print("\nClustering results summary:")
    print(clustering_df)