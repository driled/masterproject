"""
Main module for ESOL dataset dimensionality reduction and clustering analysis with ILS
and divergence metrics
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from data_loader import load_data, preprocess_data
from ils_clustering import ILS_clustering_with_optimization, ILS_clustering_with_solubility
from dim_reduction import perform_pca, perform_umap, perform_autoencoder
from pub_func.table.clustering import evaluate_embedding

from visualization import (save_embedding_csv, plot_2d_embedding,
                           plot_clustering_result, plot_silhouette_history,
                           plot_comparison_bar, save_results_summary)
from divergence_metrics import (calculate_pca_divergence, calculate_umap_divergence,
                                calculate_manifold_divergence)


def run_dimensionality_reduction(input_path, output_path):
    """
    Main function for running dimensionality reduction, clustering, and evaluation
    with Iterative Label Spreading (ILS) clustering for ESOL dataset

    Parameters:
    input_path: Input data path
    output_path: Output results path

    Returns:
    results_df: DataFrame with dimensionality reduction results
    clustering_df: DataFrame with clustering results
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load data
    X, solubility_values, solubility_bins = load_data(input_path)

    # Preprocess data
    X_scaled, _ = preprocess_data(X)

    # Initialize results containers
    all_results = []
    all_clustering_results = []

    # Run PCA analysis
    pca_results, pca_clustering_results = run_pca_analysis(X_scaled, solubility_values, solubility_bins, output_path)
    all_results.extend(pca_results)
    all_clustering_results.extend(pca_clustering_results)

    # Run UMAP analysis
    umap_results, umap_clustering_results = run_umap_analysis(X_scaled, solubility_values, solubility_bins, output_path)
    all_results.extend(umap_results)
    all_clustering_results.extend(umap_clustering_results)

    # Run Autoencoder analysis
    ae_results, ae_clustering_results = run_autoencoder_analysis(X_scaled, solubility_values, solubility_bins,
                                                                 output_path)
    all_results.extend(ae_results)
    all_clustering_results.extend(ae_clustering_results)

    # Prepare comprehensive results summary
    results_df = prepare_results_summary(all_results)

    # Prepare clustering results summary
    clustering_df = pd.DataFrame({
        'Method': [r['method'] for r in all_clustering_results],
        'Parameters': [r['params'] for r in all_clustering_results],
        'Optimal Clusters': [r['n_clusters'] for r in all_clustering_results],
        'Silhouette Score': [r['silhouette_score'] for r in all_clustering_results],
    })

    # Save results summaries
    save_results_summary(results_df, output_path, 'dimensionality_reduction_results.csv')
    save_results_summary(clustering_df, output_path, 'clustering_results.csv')

    # Evaluate with true labels (solubility categories)
    true_labels_df = evaluate_with_true_labels(X_scaled, solubility_bins, output_path)

    # Create comprehensive comparison visualizations
    create_comprehensive_visualizations(results_df, clustering_df, output_path)

    # Generate metrics correlation analysis
    correlation_analysis(results_df, output_path)

    print(f"\nAll analyses completed! Results saved to {output_path}")
    return results_df, clustering_df


def run_pca_analysis(X_scaled, solubility_values, solubility_bins, output_path):
    """
    Run PCA dimensionality reduction, clustering, and evaluation

    Parameters:
    X_scaled: Standardized feature data
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    clustering_results: List of clustering results dictionaries
    """
    results = []
    clustering_results = []

    print("\nRunning PCA dimensionality reduction...")
    for n_components in [2, 5, 10, 20]:
        print(f"  n_components = {n_components}")
        start_time = time.time()

        # Perform PCA
        X_pca, pca_model, pca_metrics = perform_pca(X_scaled, n_components)
        runtime = time.time() - start_time

        # Calculate divergence metrics
        print("  Calculating divergence metrics...")
        try:
            pca_divergence = calculate_pca_divergence(X_scaled, X_pca, pca_model)
            manifold_divergence = calculate_manifold_divergence(X_scaled, X_pca)

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
        eval_metrics = evaluate_embedding(X_scaled, X_pca, solubility_values, solubility_bins)

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
                'PCA 2D Projection', 'viridis', 'Solubility Value'
            )

            # Discrete category coloring
            if solubility_bins is not None:
                plot_2d_embedding(
                    X_pca, solubility_bins, output_path, 'pca_2d_plot_discrete.png',
                    'PCA 2D Projection (Colored by Solubility Category)', 'tab10', 'Solubility Category'
                )

        # Apply ILS clustering
        print(f"\nApplying ILS clustering to PCA (n_components={n_components}) results...")
        if solubility_bins is not None:
            print(f"\nApplying solubility-based ILS clustering...")
            cluster_labels, silhouette, history = ILS_clustering_with_solubility(X_pca, solubility_bins)
        else:
            # Use iterative optimization
            cluster_labels, silhouette, history = ILS_clustering_with_optimization(X_pca)

        # Record clustering results
        clustering_results.append({
            'method': 'PCA',
            'params': f'n_components={n_components}',
            'n_clusters': len(np.unique(cluster_labels)),
            'silhouette_score': silhouette,
            'cluster_history': history
        })

        # If 2D, create clustering visualizations
        if n_components == 2:
            # Clustering result visualization
            plot_clustering_result(
                X_pca, cluster_labels, output_path, f'pca_{n_components}d_clusters.png',
                f'PCA 2D Projection - ILS Clustering ({len(np.unique(cluster_labels))} clusters)'
            )

            # Silhouette history visualization
            plot_silhouette_history(
                history, silhouette, output_path, f'pca_{n_components}d_silhouette_history.png',
                f'PCA {n_components}D - Silhouette Coefficient vs Number of Clusters'
            )

    return results, clustering_results


def run_umap_analysis(X_scaled, solubility_values, solubility_bins, output_path):
    """
    Run UMAP dimensionality reduction, clustering, and evaluation

    Parameters:
    X_scaled: Standardized feature data
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    clustering_results: List of clustering results dictionaries
    """
    results = []
    clustering_results = []

    print("\nRunning UMAP dimensionality reduction...")
    for n_neighbors in [5, 15, 30]:
        for min_dist in [0.1, 0.5]:
            for n_components in [2, 5]:
                print(f"  n_neighbors = {n_neighbors}, min_dist = {min_dist}, n_components = {n_components}")
                start_time = time.time()

                # Perform UMAP
                X_umap, umap_model, umap_metrics = perform_umap(X_scaled, n_components, n_neighbors, min_dist)
                runtime = time.time() - start_time

                # Calculate divergence metrics
                print("  Calculating divergence metrics...")
                try:
                    umap_divergence = calculate_umap_divergence(X_scaled, X_umap, n_neighbors)
                    manifold_divergence = calculate_manifold_divergence(X_scaled, X_umap)

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
                eval_metrics = evaluate_embedding(X_scaled, X_umap, solubility_values, solubility_bins)

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
                        f'UMAP 2D Projection (n_neighbors={n_neighbors}, min_dist={min_dist})', 'viridis',
                        'Solubility Value'
                    )

                    # Discrete category coloring
                    if solubility_bins is not None:
                        plot_2d_embedding(
                            X_umap, solubility_bins, output_path,
                            f'umap_nn{n_neighbors}_md{min_dist}_2d_plot_discrete.png',
                            f'UMAP 2D Projection (n_neighbors={n_neighbors}, min_dist={min_dist}, Colored by Solubility Category)',
                            'tab10', 'Solubility Category'
                        )

                # Apply ILS clustering
                print(
                    f"\nApplying ILS clustering to UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}) results...")
                if solubility_bins is not None:
                    print(f"\nApplying solubility-based ILS clustering...")
                    cluster_labels, silhouette, history = ILS_clustering_with_solubility(X_umap, solubility_bins)
                else:
                    # Use iterative optimization
                    cluster_labels, silhouette, history = ILS_clustering_with_optimization(X_umap)

                # Record clustering results
                clustering_results.append({
                    'method': 'UMAP',
                    'params': f'n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}',
                    'n_clusters': len(np.unique(cluster_labels)),
                    'silhouette_score': silhouette,
                    'cluster_history': history
                })

                # If 2D, create clustering visualizations
                if n_components == 2:
                    # Clustering result visualization
                    plot_clustering_result(
                        X_umap, cluster_labels, output_path,
                        f'umap_nn{n_neighbors}_md{min_dist}_{n_components}d_clusters.png',
                        f'UMAP 2D Projection (n_neighbors={n_neighbors}, min_dist={min_dist}) - ILS Clustering ({len(np.unique(cluster_labels))} clusters)'
                    )

                    # Silhouette history visualization
                    plot_silhouette_history(
                        history, silhouette, output_path,
                        f'umap_nn{n_neighbors}_md{min_dist}_{n_components}d_silhouette_history.png',
                        f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) {n_components}D - Silhouette Coefficient vs Number of Clusters'
                    )

    return results, clustering_results


def run_autoencoder_analysis(X_scaled, solubility_values, solubility_bins, output_path):
    """
    Run autoencoder dimensionality reduction, clustering, and evaluation

    Parameters:
    X_scaled: Standardized feature data
    solubility_values: Continuous solubility values
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    results: List of results dictionaries
    clustering_results: List of clustering results dictionaries
    """
    results = []
    clustering_results = []

    print("\nRunning autoencoder dimensionality reduction...")

    # Try different encoding dimensions and intermediate layer sizes
    for encoding_dim in [2, 5, 10, 20]:
        for intermediate_dim in [128, 256, 512]:
            print(f"  encoding_dim = {encoding_dim}, intermediate_dim = {intermediate_dim}")
            start_time = time.time()

            # Perform autoencoder dimensionality reduction
            X_ae, encoder, ae_metrics = perform_autoencoder(X_scaled, encoding_dim, intermediate_dim)
            runtime = time.time() - start_time

            # Calculate divergence metrics
            print("  Calculating divergence metrics...")
            try:
                # The decoder is needed for autoencoder divergence
                # We need to extract it from the perform_autoencoder result
                # For simplicity, we'll use manifold divergence which doesn't need the decoder
                manifold_divergence = calculate_manifold_divergence(X_scaled, X_ae)

                # Add divergence metrics to ae_metrics
                ae_metrics['manifold_divergence'] = manifold_divergence

                print(
                    f"  Manifold Divergence: {manifold_divergence:.4f}" if manifold_divergence is not None else "  Manifold Divergence: N/A")
            except Exception as e:
                print(f"  Error calculating divergence metrics: {str(e)}")

            # Save reduction results
            save_embedding_csv(X_ae, output_path, f"ae_ed{encoding_dim}_id{intermediate_dim}.csv")

            # Evaluate reduction quality
            eval_metrics = evaluate_embedding(X_scaled, X_ae, solubility_values, solubility_bins)

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
                    f'AE 2D Projection (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim})',
                    'viridis', 'Solubility Value'
                )

                # Discrete category coloring
                if solubility_bins is not None:
                    plot_2d_embedding(
                        X_ae, solubility_bins, output_path,
                        f'ae_ed{encoding_dim}_id{intermediate_dim}_2d_plot_discrete.png',
                        f'AE 2D Projection (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}, Colored by Solubility Category)',
                        'tab10', 'Solubility Category'
                    )

            # Apply ILS clustering
            print(
                f"\nApplying ILS clustering to autoencoder results (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim})...")
            if solubility_bins is not None:
                print(f"\nApplying solubility-based ILS clustering...")
                cluster_labels, silhouette, history = ILS_clustering_with_solubility(X_ae, solubility_bins)
            else:
                # Use iterative optimization
                cluster_labels, silhouette, history = ILS_clustering_with_optimization(X_ae)

            # Record clustering results
            clustering_results.append({
                'method': 'Autoencoder',
                'params': f'encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}',
                'n_clusters': len(np.unique(cluster_labels)),
                'silhouette_score': silhouette,
                'cluster_history': history
            })

            # If 2D, create clustering visualizations
            if encoding_dim == 2:
                # Clustering result visualization
                plot_clustering_result(
                    X_ae, cluster_labels, output_path, f'ae_ed{encoding_dim}_id{intermediate_dim}_clusters.png',
                    f'AE 2D Projection (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}) - ILS Clustering ({len(np.unique(cluster_labels))} clusters)'
                )

                # Silhouette history visualization
                plot_silhouette_history(
                    history, silhouette, output_path,
                    f'ae_ed{encoding_dim}_id{intermediate_dim}_silhouette_history.png',
                    f'AE (encoding_dim={encoding_dim}, intermediate_dim={intermediate_dim}) - Silhouette Coefficient vs Number of Clusters'
                )

    return results, clustering_results


def evaluate_with_true_labels(X_scaled, solubility_bins, output_path):
    """
    Calculate silhouette scores using true solubility categories

    Parameters:
    X_scaled: Standardized feature data
    solubility_bins: Discretized solubility categories
    output_path: Output directory path

    Returns:
    true_labels_df: DataFrame with silhouette scores for true labels
    """
    if solubility_bins is None:
        return None

    silhouette_with_true_labels = []

    # PCA dimensionality reduction
    for n_components in [2, 5, 10, 20]:
        X_reduced = perform_pca(X_scaled, n_components)[0]
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
                X_reduced = perform_umap(X_scaled, n_components, n_neighbors, min_dist)[0]
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
        clustering_df, 'Dimensionality Reduction Method', 'Silhouette Score',
        'ILS Clustering Silhouette Score Comparison', output_path,
        'clustering_silhouette_comparison.png'
    )


if __name__ == "__main__":
    # Set input and output paths
    input_path = r"D:\materproject\all-reps\ESOL\ESOL-table"
    output_path = r"D:\materproject\single-rep-rd\table\ESOL-ILS"

    # Run main function
    results_df, clustering_df = run_dimensionality_reduction(input_path, output_path)

    # Print results summary
    print("\nDimensionality reduction results summary:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df)

    print("\nClustering results summary:")
    print(clustering_df)