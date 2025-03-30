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