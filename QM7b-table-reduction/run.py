"""
QM7 dataset dimensionality reduction and iterative label spilling clustering analysis
"""

import os
import sys


def setup_paths():
    """Set up module paths"""
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add current directory to Python path
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    print(f"Added {current_dir} to Python path")


if __name__ == "__main__":
    # Set up paths
    setup_paths()

    # Import main module
    from main import run_dimensionality_reduction

    # Set input and output paths
    input_path = r"D:\materproject\all-reps\QM7b\QM7-table"
    output_path = r"D:\materproject\single-rep-rd\table\QM7"

    # Allow modification of paths via command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Run analysis
    results_df, clustering_df = run_dimensionality_reduction(input_path, output_path)

    print("\nAnalysis completed!")