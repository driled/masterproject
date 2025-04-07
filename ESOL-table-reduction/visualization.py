import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_embedding_csv(X_embedded, output_path, filename):
    """
    Save dimensionality reduction results to CSV file

    Parameters:
    X_embedded: Reduced dimension data
    output_path: Output directory path
    filename: Output filename
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    pd.DataFrame(X_embedded).to_csv(output_file, index=False)
    print(f"Reduction results saved to: {output_file}")


def plot_2d_embedding(X_embedded, values, output_path, filename, title, colormap='viridis', label='Value'):
    """
    Plot 2D dimensionality reduction results as a scatter plot

    Parameters:
    X_embedded: 2D reduction data
    values: Values for coloring
    output_path: Output directory path
    filename: Output filename
    title: Chart title
    colormap: Color map
    label: Colorbar label
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=values, cmap=colormap, alpha=0.7)
    plt.colorbar(scatter, label=label)
    plt.title(title)

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Image saved to: {output_file}")


def plot_clustering_result(X_embedded, cluster_labels, output_path, filename, title):
    """
    Plot clustering results as a scatter plot

    Parameters:
    X_embedded: 2D reduction data
    cluster_labels: Cluster labels
    output_path: Output directory path
    filename: Output filename
    title: Chart title
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Clustering result image saved to: {output_file}")


def plot_silhouette_history(history, best_silhouette, output_path, filename, title):
    """
    Plot silhouette coefficient vs. number of clusters

    Parameters:
    history: History record containing cluster numbers and silhouette coefficients
    best_silhouette: Best silhouette coefficient
    output_path: Output directory path
    filename: Output filename
    title: Chart title
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot([item['n_clusters'] for item in history],
             [item['silhouette'] for item in history],
             marker='o')
    plt.axhline(y=best_silhouette, color='r', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title(title)
    plt.grid(True)

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Silhouette history image saved to: {output_file}")


def plot_divergence_comparison(results_df, output_path):
    """
    Create comparison visualizations for divergence metrics across all methods

    Parameters:
    results_df: DataFrame with dimensionality reduction results
    output_path: Output directory path
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_path, exist_ok=True)

    # Extract divergence metrics columns
    divergence_cols = [
        'PCA Divergence', 't-SNE Divergence', 'UMAP Divergence',
        'KL Divergence', 'Autoencoder Divergence', 'Manifold Divergence'
    ]

    # Prepare data for divergence comparison
    plot_data = []

    for _, row in results_df.iterrows():
        method = row['Method']
        params = row['Parameters']

        # For each divergence metric that exists
        for col in divergence_cols:
            if pd.notnull(row[col]):
                plot_data.append({
                    'Method': method,
                    'Parameters': params,
                    'Divergence Type': col,
                    'Value': row[col]
                })

    if not plot_data:
        print("No divergence metrics available for visualization")
        return

    # Create DataFrame from collected data
    plot_df = pd.DataFrame(plot_data)

    # 1. Overall comparison of divergence metrics by method
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='Divergence Type', y='Value', hue='Method', data=plot_df)
    plt.title('Comparison of Divergence Metrics Across Methods')
    plt.xlabel('Divergence Metric Type')
    plt.ylabel('Divergence Value')
    plt.yscale('log')  # Log scale for better visibility of differences
    plt.xticks(rotation=45)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'divergence_comparison_boxplot.png'))
    plt.close()

    # 2. Method-specific divergence by parameters
    for method in plot_df['Method'].unique():
        method_df = plot_df[plot_df['Method'] == method]

        # Only create plot if we have data
        if len(method_df) > 0:
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='Parameters', y='Value', hue='Divergence Type', data=method_df)
            plt.title(f'Divergence Metrics for {method} by Parameters')
            plt.xlabel('Parameters')
            plt.ylabel('Divergence Value')
            plt.xticks(rotation=90)
            plt.legend(title='Divergence Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{method.lower()}_divergence_by_params.png'))
            plt.close()

    # 3. Correlation between divergence and other quality metrics
    quality_metrics = ['Neighbor Preservation', 'Silhouette Score', 'Trustworthiness',
                       'Continuity', 'Reconstruction Error']

    # Prepare data for correlation analysis
    corr_data = results_df.copy()

    # Only include rows with at least one divergence metric
    corr_data = corr_data.loc[corr_data[divergence_cols].notna().any(axis=1)]

    if len(corr_data) > 1:  # Need at least 2 points for correlation
        # Calculate correlation for each quality metric against each divergence metric
        corr_results = []

        for quality_metric in quality_metrics:
            for div_metric in divergence_cols:
                # Filter rows with both metrics available
                valid_data = corr_data[[quality_metric, div_metric]].dropna()

                if len(valid_data) > 1:  # Need at least 2 points for correlation
                    corr = valid_data[quality_metric].corr(valid_data[div_metric])
                    corr_results.append({
                        'Quality Metric': quality_metric,
                        'Divergence Metric': div_metric,
                        'Correlation': corr
                    })

        if corr_results:
            corr_df = pd.DataFrame(corr_results)

            # Create a pivot table for the heatmap
            corr_pivot = corr_df.pivot(index='Quality Metric',
                                       columns='Divergence Metric',
                                       values='Correlation')

            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_pivot, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Correlation between Quality Metrics and Divergence Metrics')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'divergence_quality_correlation.png'))
            plt.close()

            # Save correlation results
            corr_df.to_csv(os.path.join(output_path, 'divergence_quality_correlation.csv'), index=False)

    # 4. Scatter plots for divergence vs. other key metrics
    for quality_metric in ['Neighbor Preservation', 'Silhouette Score', 'Trustworthiness']:
        for div_metric in divergence_cols:
            valid_data = corr_data[[quality_metric, div_metric, 'Method']].dropna()

            if len(valid_data) > 2:  # Need at least 3 points for a meaningful scatter plot
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x=div_metric, y=quality_metric, hue='Method', data=valid_data, s=100)

                # Add trend line
                sns.regplot(x=div_metric, y=quality_metric, data=valid_data,
                            scatter=False, ci=None, line_kws={"color": "black", "linestyle": "--"})

                plt.title(f'{quality_metric} vs {div_metric}')
                plt.xlabel(div_metric)
                plt.ylabel(quality_metric)
                plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                # Clean filename
                filename = f'{quality_metric.lower().replace(" ", "_")}_vs_{div_metric.lower().replace(" ", "_")}.png'
                plt.savefig(os.path.join(output_path, filename))
                plt.close()

    print(f"Divergence visualization complete. Output saved to {output_path}")


def plot_vae_kl_divergence_history(all_results, output_path):
    """
    Create visualizations of KL divergence history for VAE training

    Parameters:
    all_results: List of results dictionaries from all dimensionality reduction methods
    output_path: Output directory path
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_path, exist_ok=True)

    # Filter VAE results that have KL divergence history
    vae_results = [r for r in all_results if r['method'] == 'VAE' and 'kl_divergence_history' in r['metrics']]

    if not vae_results:
        print("No VAE KL divergence history available for visualization")
        return

    # Plot KL divergence history for each VAE configuration
    for result in vae_results:
        kl_history = result['metrics']['kl_divergence_history']
        params = result['params']

        if kl_history and len(kl_history) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(kl_history) + 1), kl_history, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('KL Divergence')
            plt.title(f'KL Divergence During VAE Training\n{params}')
            plt.grid(True)

            # Clean filename
            filename = f'vae_kl_history_{params.replace("=", "_").replace(", ", "_").replace(" ", "")}.png'
            plt.savefig(os.path.join(output_path, filename))
            plt.close()

    # Comparative plot of all VAE configurations
    if len(vae_results) > 1:
        plt.figure(figsize=(12, 8))

        for result in vae_results:
            kl_history = result['metrics']['kl_divergence_history']
            params = result['params']

            if kl_history and len(kl_history) > 1:
                plt.plot(range(1, len(kl_history) + 1), kl_history, marker='o', label=params)

        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Comparison During VAE Training')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'vae_kl_history_comparison.png'))
        plt.close()

    print(f"VAE KL divergence history visualization complete. Output saved to {output_path}")


def plot_comparison_bar(data, x_label, y_label, title, output_path, filename):
    """
    Plot comparison bar chart

    Parameters:
    data: DataFrame containing comparison data
    x_label: X-axis label
    y_label: Y-axis label
    title: Chart title
    output_path: Output directory path
    filename: Output filename
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(14, 7))

    # Get unique method names
    methods = data['Method'].unique()

    bar_width = 0.15
    index = np.arange(len(methods))
    param_groups = data.groupby(['Method', 'Parameters'])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    # Create bar chart for each method-parameter combination
    for method in methods:
        method_params = data[data['Method'] == method]['Parameters'].unique()
        for i, params in enumerate(method_params):
            subset = data[(data['Method'] == method) & (data['Parameters'] == params)]
            if len(subset) > 0 and y_label in subset.columns:
                # Ensure there are values to plot
                if not pd.isna(subset[y_label].values[0]):
                    pos = index[np.where(methods == method)[0][0]] + (i - len(method_params) / 2 + 0.5) * bar_width
                    plt.bar(pos, subset[y_label].values[0], bar_width,
                            label=f"{method}: {params}", color=colors[color_idx % len(colors)])
            color_idx += 1

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index, methods, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to: {output_file}")


def save_results_summary(results_df, output_path, filename):
    """
    Save results summary to CSV file

    Parameters:
    results_df: DataFrame containing results
    output_path: Output directory path
    filename: Output filename
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    results_df.to_csv(output_file, index=False)
    print(f"Results summary saved to: {output_file}")