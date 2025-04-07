"""
Analysis script for evaluating divergence metrics across dimensionality reduction methods
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_divergence_results(results_path):
    """
    Analyze divergence metrics from dimensionality reduction results

    Parameters:
    results_path: Path to the results directory containing the results CSV
    """
    # Load results
    results_file = os.path.join(results_path, 'dimensionality_reduction_results.csv')

    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    results_df = pd.read_csv(results_file)

    # Identify divergence columns
    divergence_cols = [col for col in results_df.columns if 'Divergence' in col]

    if not divergence_cols:
        print("No divergence metrics found in results")
        return

    print(f"Found divergence metrics: {', '.join(divergence_cols)}")

    # Create analysis directory
    analysis_dir = os.path.join(results_path, 'divergence_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Basic statistics for each divergence metric
    divergence_stats = {}

    for col in divergence_cols:
        valid_data = results_df[col].dropna()

        if len(valid_data) > 0:
            stats = {
                'count': len(valid_data),
                'min': valid_data.min(),
                'max': valid_data.max(),
                'mean': valid_data.mean(),
                'median': valid_data.median(),
                'std': valid_data.std()
            }
            divergence_stats[col] = stats

    # Save statistics to CSV
    stats_df = pd.DataFrame(divergence_stats).T
    stats_df.to_csv(os.path.join(analysis_dir, 'divergence_statistics.csv'))

    print("Divergence statistics summary:")
    print(stats_df)

    # 2. Rank methods by each divergence metric (lower is better)
    ranking_results = {}

    for col in divergence_cols:
        # Filter rows with this divergence metric
        valid_data = results_df[['Method', 'Parameters', col]].dropna()

        if len(valid_data) > 0:
            # Sort by divergence (ascending)
            ranked = valid_data.sort_values(by=col)

            # Save top 5 configurations
            top5 = ranked.head(5)
            ranking_results[col] = top5

            # Save to CSV
            ranked.to_csv(os.path.join(analysis_dir, f'{col.replace(" ", "_").lower()}_ranking.csv'), index=False)

    # 3. Create divergence heatmap by method and parameters
    for col in divergence_cols:
        valid_data = results_df[['Method', 'Parameters', col]].dropna()

        if len(valid_data) > 0:
            # Create a pivot table for the methods and their divergence values
            methods = valid_data['Method'].unique()

            for method in methods:
                method_data = valid_data[valid_data['Method'] == method]

                if len(method_data) > 1:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Parameters', y=col, data=method_data)
                    plt.title(f'{col} for {method}')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(os.path.join(analysis_dir, f'{method.lower()}_{col.replace(" ", "_").lower()}.png'))
                    plt.close()

    # 4. Cross-correlation between divergence metrics
    divergence_correlation = results_df[divergence_cols].corr()
    divergence_correlation.to_csv(os.path.join(analysis_dir, 'divergence_cross_correlation.csv'))

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(divergence_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Divergence Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'divergence_correlation_heatmap.png'))
    plt.close()

    # 5. Correlation with other quality metrics
    quality_metrics = ['Neighbor Preservation', 'Silhouette Score', 'Trustworthiness',
                       'Continuity', 'Reconstruction Error']

    # Filter columns that exist
    existing_quality_metrics = [col for col in quality_metrics if col in results_df.columns]

    if existing_quality_metrics:
        # Calculate correlation matrix
        quality_correlation = results_df[existing_quality_metrics + divergence_cols].corr()
        quality_correlation.to_csv(os.path.join(analysis_dir, 'quality_divergence_correlation.csv'))

        # Create subset of correlation matrix
        subset_corr = quality_correlation.loc[existing_quality_metrics, divergence_cols]

        plt.figure(figsize=(12, 8))
        sns.heatmap(subset_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Between Quality Metrics and Divergence Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'quality_divergence_correlation_heatmap.png'))
        plt.close()

    # 6. Relationship between dimensionality and divergence
    for method in results_df['Method'].unique():
        method_data = results_df[results_df['Method'] == method].copy()

        # Extract dimensionality from parameters if possible
        if 'n_components=' in method_data['Parameters'].iloc[0] or 'latent_dim=' in method_data['Parameters'].iloc[0]:
            # Extract dimensionality
            method_data['Dimensionality'] = method_data['Parameters'].apply(
                lambda x: int(x.split('n_components=')[1].split(',')[0])
                if 'n_components=' in x
                else int(x.split('latent_dim=')[1].split(',')[0])
                if 'latent_dim=' in x
                else np.nan
            )

            # Plot dimensionality vs divergence for each divergence metric
            for col in divergence_cols:
                valid_data = method_data[['Dimensionality', col]].dropna()

                if len(valid_data) > 1:
                    plt.figure(figsize=(8, 6))
                    sns.lineplot(x='Dimensionality', y=col, data=valid_data, marker='o')
                    plt.title(f'{col} vs Dimensionality for {method}')
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(analysis_dir, f'{method.lower()}_dim_vs_{col.replace(" ", "_").lower()}.png'))
                    plt.close()

    print(f"Divergence analysis complete. Results saved to {analysis_dir}")


if __name__ == "__main__":
    # Use command line argument for results path, or use default
    results_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\materproject\single-rep-rd\table\ESOL"

    print(f"Analyzing divergence metrics from: {results_path}")
    analyze_divergence_results(results_path)