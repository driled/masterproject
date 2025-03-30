"""
Main module for ESOL dataset dimensionality reduction and clustering analysis
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from data_loader import load_data, preprocess_data
from clustering import iterative_label_spilling, evaluate_embedding
from dim_reduction import (perform_pca, perform_tsne,
                           perform_umap, perform_vae)
from visualization import (save_embedding_csv, plot_2d_embedding,
                           plot_clustering_result, plot_silhouette_history,
                           plot_comparison_bar, save_results_summary)


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

        # Apply iterative label spilling clustering
        print(f"\nApplying iterative label spilling clustering to PCA (n_components={n_components}) results...")
        cluster_labels, silhouette, history = iterative_label_spilling(X_pca)

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
                f'PCA 2D Projection - Iterative Label Spilling Clustering ({len(np.unique(cluster_labels))} clusters)'
            )

            # Silhouette history visualization
            plot_silhouette_history(
                history, silhouette, output_path, f'pca_{n_components}d_silhouette_history.png',
                f'PCA {n_components}D - Silhouette Coefficient vs Number of Clusters'
            )

    return results, clustering_results


def run_tsne_analysis(X_scaled, solubility_values, solubility_bins, output_path):
    """
    Run t-SNE dimensionality reduction, clustering, and evaluation

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

    print("\nRunning t-SNE dimensionality reduction...")
    for perplexity in [5, 30, 50]:
        for n_components in [2, 3]:
            print(f"  perplexity = {perplexity}, n_components = {n_components}")
            start_time = time.time()

            # Perform t-SNE
            X_tsne, _, tsne_metrics = perform_tsne(X_scaled, n_components, perplexity)
            runtime = time.time() - start_time

            # Save reduction results
            save_embedding_csv(X_tsne, output_path, f"tsne_perp{perplexity}_{n_components}d.csv")

            # Evaluate reduction quality
            eval_metrics = evaluate_embedding(X_scaled, X_tsne, solubility_values, solubility_bins)

            # Merge all metrics
            all_metrics = {**eval_metrics, **tsne_metrics}

            # Record results
            results.append({
                'method': 't-SNE',
                'params': f'perplexity={perplexity}, n_components={n_components}',
                'runtime': runtime,
                'metrics': all_metrics
            })

            # If 2D, create visualizations
            if n_components == 2:
                # Continuous value coloring
                plot_2d_embedding(
                    X_tsne, solubility_values, output_path, f'tsne_perp{perplexity}_2d_plot.png',
                    f't-SNE 2D Projection (perplexity={perplexity})', 'viridis', 'Solubility Value'
                )

                # Discrete category coloring
                if solubility_bins is not None:
                    plot_2d_embedding(
                        X_tsne, solubility_bins, output_path, f'tsne_perp{perplexity}_2d_plot_discrete.png',
                        f't-SNE 2D Projection (perplexity={perplexity}, Colored by Solubility Category)', 'tab10',
                        'Solubility Category'
                    )

            # Apply iterative label spilling clustering
            print(
                f"\nApplying iterative label spilling clustering to t-SNE (perplexity={perplexity}, n_components={n_components}) results...")
            cluster_labels, silhouette, history = iterative_label_spilling(X_tsne)

            # Record clustering results
            clustering_results.append({
                'method': 't-SNE',
                'params': f'perplexity={perplexity}, n_components={n_components}',
                'n_clusters': len(np.unique(cluster_labels)),
                'silhouette_score': silhouette,
                'cluster_history': history
            })

            # If 2D, create clustering visualizations
            if n_components == 2:
                # Clustering result visualization
                plot_clustering_result(
                    X_tsne, cluster_labels, output_path, f'tsne_perp{perplexity}_{n_components}d_clusters.png',
                    f't-SNE 2D Projection (perplexity={perplexity}) - Iterative Label Spilling Clustering ({len(np.unique(cluster_labels))} clusters)'
                )

                # Silhouette history visualization
                plot_silhouette_history(
                    history, silhouette, output_path, f'tsne_perp{perplexity}_{n_components}d_silhouette_history.png',
                    f't-SNE (perplexity={perplexity}) {n_components}D - Silhouette Coefficient vs Number of Clusters'
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
                X_umap, _, umap_metrics = perform_umap(X_scaled, n_components, n_neighbors, min_dist)
                runtime = time.time() - start_time

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

                # Apply iterative label spilling clustering
                print(
                    f"\nApplying iterative label spilling clustering to UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}) results...")
                cluster_labels, silhouette, history = iterative_label_spilling(X_umap)

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
                        f'UMAP 2D Projection (n_neighbors={n_neighbors}, min_dist={min_dist}) - Iterative Label Spilling Clustering ({len(np.unique(cluster_labels))} clusters)'
                    )

                    # Silhouette history visualization
                    plot_silhouette_history(
                        history, silhouette, output_path,
                        f'umap_nn{n_neighbors}_md{min_dist}_{n_components}d_silhouette_history.png',
                        f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) {n_components}D - Silhouette Coefficient vs Number of Clusters'
                    )

    return results, clustering_results


def run_vae_analysis(X_scaled, solubility_values, solubility_bins, output_path):
    """
    Run VAE dimensionality reduction, clustering, and evaluation

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

    print("\nRunning VAE dimensionality reduction...")
    for latent_dim in [2, 5, 10]:
        for intermediate_dim in [128, 256]:
            print(f"  latent_dim = {latent_dim}, intermediate_dim = {intermediate_dim}")
            start_time = time.time()

            # Perform VAE
            X_vae, encoder, vae_metrics = perform_vae(X_scaled, latent_dim, intermediate_dim)
            runtime = time.time() - start_time

            # Save reduction results
            save_embedding_csv(X_vae, output_path, f"vae_ld{latent_dim}_id{intermediate_dim}.csv")

            # Evaluate reduction quality
            eval_metrics = evaluate_embedding(X_scaled, X_vae, solubility_values, solubility_bins)

            # Merge all metrics
            all_metrics = {**eval_metrics, **vae_metrics}

            # Record results
            results.append({
                'method': 'VAE',
                'params': f'latent_dim={latent_dim}, intermediate_dim={intermediate_dim}',
                'runtime': runtime,
                'metrics': all_metrics
            })

            # If 2D, create visualizations
            if latent_dim == 2:
                # Continuous value coloring
                plot_2d_embedding(
                    X_vae, solubility_values, output_path, f'vae_ld{latent_dim}_id{intermediate_dim}_2d_plot.png',
                    f'VAE 2D Projection (latent_dim={latent_dim}, intermediate_dim={intermediate_dim})', 'viridis',
                    'Solubility Value'
                )

                # Discrete category coloring
                if solubility_bins is not None:
                    plot_2d_embedding(
                        X_vae, solubility_bins, output_path,
                        f'vae_ld{latent_dim}_id{intermediate_dim}_2d_plot_discrete.png',
                        f'VAE 2D Projection (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}, Colored by Solubility Category)',
                        'tab10', 'Solubility Category'
                    )

            # Apply iterative label spilling clustering
            print(
                f"\nApplying iterative label spilling clustering to VAE (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}) results...")
            cluster_labels, silhouette, history = iterative_label_spilling(X_vae)

            # Record clustering results
            clustering_results.append({
                'method': 'VAE',
                'params': f'latent_dim={latent_dim}, intermediate_dim={intermediate_dim}',
                'n_clusters': len(np.unique(cluster_labels)),
                'silhouette_score': silhouette,
                'cluster_history': history
            })

            # If 2D, create clustering visualizations
            if latent_dim == 2:
                # Clustering result visualization
                plot_clustering_result(
                    X_vae, cluster_labels, output_path, f'vae_ld{latent_dim}_id{intermediate_dim}_clusters.png',
                    f'VAE 2D Projection (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}) - Iterative Label Spilling Clustering ({len(np.unique(cluster_labels))} clusters)'
                )

                # Silhouette history visualization
                plot_silhouette_history(
                    history, silhouette, output_path, f'vae_ld{latent_dim}_id{intermediate_dim}_silhouette_history.png',
                    f'VAE (latent_dim={latent_dim}, intermediate_dim={intermediate_dim}) - Silhouette Coefficient vs Number of Clusters'
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

    # t-SNE dimensionality reduction
    for perplexity in [5, 30, 50]:
        for n_components in [2, 3]:
            X_reduced = perform_tsne(X_scaled, n_components, perplexity)[0]
            try:
                s_score = silhouette_score(X_reduced, solubility_bins)
                silhouette_with_true_labels.append({
                    'Method': 't-SNE',
                    'Parameters': f'perplexity={perplexity}, n_components={n_components}',
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

    # VAE dimensionality reduction
    for latent_dim in [2, 5, 10]:
        for intermediate_dim in [128, 256]:
            X_reduced = perform_vae(X_scaled, latent_dim, intermediate_dim)[0]
            try:
                s_score = silhouette_score(X_reduced, solubility_bins)
                silhouette_with_true_labels.append({
                    'Method': 'VAE',
                    'Parameters': f'latent_dim={latent_dim}, intermediate_dim={intermediate_dim}',
                    'Silhouette Score (True Labels)': s_score
                })
            except:
                pass

    # Create DataFrame
    true_labels_df = pd.DataFrame(silhouette_with_true_labels)

    # Save results
    save_results_summary(true_labels_df, output_path, 'silhouette_with_true_labels.csv')

    return true_labels_df


def create_comparison_visualizations(results_df, clustering_df, output_path):
    """
    Create comparative visualizations of dimensionality reduction and clustering results

    Parameters:
    results_df: DataFrame with dimensionality reduction results
    clustering_df: DataFrame with clustering results
    output_path: Output directory path
    """
    # 1. Runtime comparison
    plot_comparison_bar(
        results_df, 'Dimensionality Reduction Method', 'Runtime (s)',
        'Runtime Comparison of Dimensionality Reduction Methods', output_path, 'runtime_comparison.png'
    )

    # 2. Neighbor preservation comparison
    plot_comparison_bar(
        results_df, 'Dimensionality Reduction Method', 'Neighbor Preservation',
        'Neighbor Preservation Comparison of Dimensionality Reduction Methods', output_path,
        'neighbor_preservation_comparison.png'
    )

    # 3. Silhouette score comparison (if data exists)
    if any(~results_df['Silhouette Score'].isna()):
        plot_comparison_bar(
            results_df, 'Dimensionality Reduction Method', 'Silhouette Score',
            'Silhouette Score Comparison of Dimensionality Reduction Methods', output_path,
            'silhouette_score_comparison.png'
        )

    # 4. Clustering silhouette score comparison
    plot_comparison_bar(
        clustering_df, 'Dimensionality Reduction Method', 'Silhouette Score',
        'Iterative Label Spilling Clustering Silhouette Score Comparison', output_path,
        'clustering_silhouette_comparison.png'
    )


def run_dimensionality_reduction(input_path, output_path):
    """
    Main function for running dimensionality reduction, clustering, and evaluation

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

    # Run t-SNE analysis
    tsne_results, tsne_clustering_results = run_tsne_analysis(X_scaled, solubility_values, solubility_bins, output_path)
    all_results.extend(tsne_results)
    all_clustering_results.extend(tsne_clustering_results)

    # Run UMAP analysis
    umap_results, umap_clustering_results = run_umap_analysis(X_scaled, solubility_values, solubility_bins, output_path)
    all_results.extend(umap_results)
    all_clustering_results.extend(umap_clustering_results)

    # Run VAE analysis
    vae_results, vae_clustering_results = run_vae_analysis(X_scaled, solubility_values, solubility_bins, output_path)
    all_results.extend(vae_results)
    all_clustering_results.extend(vae_clustering_results)

    # Prepare results summary
    results_df = pd.DataFrame({
        'Method': [r['method'] for r in all_results],
        'Parameters': [r['params'] for r in all_results],
        'Runtime (s)': [r['runtime'] for r in all_results],
        'Neighbor Preservation': [r['metrics'].get('neighbor_preservation', None) for r in all_results],
        'Silhouette Score': [r['metrics'].get('silhouette_score', None) for r in all_results],
        'Variance Explained': [r['metrics'].get('variance_explained', None) for r in all_results],
        'Reconstruction Error': [r['metrics'].get('reconstruction_error', None) for r in all_results],
    })

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

    # Evaluate with true labels
    true_labels_df = evaluate_with_true_labels(X_scaled, solubility_bins, output_path)

    # Create comparison visualizations
    create_comparison_visualizations(results_df, clustering_df, output_path)

    print(f"\nAll analyses completed! Results saved to {output_path}")
    return results_df, clustering_df


if __name__ == "__main__":
    # Set input and output paths
    input_path = r"D:\materproject\all-reps\ESOL\ESOL-table"
    output_path = r"D:\materproject\single-rep-rd\table\ESOL"

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