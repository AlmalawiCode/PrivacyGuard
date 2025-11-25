#!/usr/bin/env python3
"""
Script to generate comparison figures for synthetic data generation methods.
Creates three PDF figures:
1. Total Generation Time comparison
2. Training Time comparison
3. Testing Time comparison
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define base paths
BASE_DIR = Path("/home/abdul/claudeProjects/VWC-SyntheticData")
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "output" / "averaged_classification_results"
OUTPUT_DIR = Path(__file__).parent  # Save in same directory as script

# Define methods and datasets
METHODS = ["Equal_Width_Binning", "k_anonymity", "kmeans", "LaplaceDP", "PrivacyGuard"]
DATASETS = ["bot_loT", "CICIoT2023", "Edge-IIoTset", "MQTTset"]

# Method display names for better readability
METHOD_DISPLAY_NAMES = {
    "Equal_Width_Binning": "Equal Width Binning",
    "k_anonymity": "k-Anonymity",
    "kmeans": "K-Means",
    "LaplaceDP": "Laplace DP",
    "PrivacyGuard": "PrivacyGuard"
}

# Dataset display names (code name -> paper name)
DATASET_DISPLAY_NAMES = {
    "bot_loT": "N-BaIoT",
    "Bot-IoT": "N-BaIoT",
    "CICIoT2023": "CICIoT2023",
    "Edge-IIoTset": "Edge-IIoTset",
    "MQTTset": "MQTTset"
}

# Consistent color mapping for each method across all figures
METHOD_COLORS = {
    "Equal Width Binning": "#2ecc71",  # Green
    "k-Anonymity": "#3498db",          # Blue
    "K-Means": "#e74c3c",              # Red
    "Laplace DP": "#f39c12",           # Orange
    "PrivacyGuard": "#9b59b6",         # Purple
    "Original": "#34495e"              # Dark gray (if present)
}


def extract_generation_time(summary_file):
    """Extract total generation time from a generation summary file."""
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        # Look for "Total Generation Time: X.XX seconds"
        match = re.search(r'Total Generation Time:\s*([\d.]+)\s*seconds', content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
    return None


def collect_generation_times():
    """Collect generation times for all methods and datasets."""
    data = []

    for method in METHODS:
        for dataset in DATASETS:
            summary_file = DATASETS_DIR / method / f"{dataset}_generation_summary.txt"
            if summary_file.exists():
                gen_time = extract_generation_time(summary_file)
                if gen_time is not None:
                    data.append({
                        'Method': METHOD_DISPLAY_NAMES.get(method, method),
                        'Dataset': dataset,
                        'Generation_Time': gen_time
                    })

    return pd.DataFrame(data)


def collect_training_testing_times():
    """Collect training and testing times from averaged classification results."""
    training_data = []
    testing_data = []

    for dataset in DATASETS:
        result_file = RESULTS_DIR / f"averaged_results_{dataset}.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)

            # Filter for training time
            train_df = df[df['Metric'] == 'Training Time (ns)']
            for _, row in train_df.iterrows():
                method = row['Dataset']
                classifier = row['Classifier']
                time_ns = row['Value']
                time_sec = time_ns / 1e9  # Convert nanoseconds to seconds
                training_data.append({
                    'Method': METHOD_DISPLAY_NAMES.get(method, method),
                    'Dataset': dataset,
                    'Classifier': classifier,
                    'Training_Time': time_sec
                })

            # Filter for testing time
            test_df = df[df['Metric'] == 'Testing Time (ns)']
            for _, row in test_df.iterrows():
                method = row['Dataset']
                classifier = row['Classifier']
                time_ns = row['Value']
                time_sec = time_ns / 1e9  # Convert nanoseconds to seconds
                testing_data.append({
                    'Method': METHOD_DISPLAY_NAMES.get(method, method),
                    'Dataset': dataset,
                    'Classifier': classifier,
                    'Testing_Time': time_sec
                })

    return pd.DataFrame(training_data), pd.DataFrame(testing_data)


def plot_generation_time(df, output_dir):
    """Create separate bar charts for generation time comparison using relative values."""
    datasets = sorted(df['Dataset'].unique())

    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = df[df['Dataset'] == dataset].copy()

        # Sort by generation time (ascending order - smallest to largest)
        dataset_df = dataset_df.sort_values('Generation_Time')

        # Get methods and their times
        methods = dataset_df['Method'].values
        times = dataset_df['Generation_Time'].values

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        x = np.arange(len(methods))
        bars = ax.bar(x, times, width=0.6, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Color bars with consistent colors for each method
        for bar, method in zip(bars, methods):
            bar.set_facecolor(METHOD_COLORS.get(method, '#95a5a6'))

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Set y-axis limits to accommodate labels (with extra space on top)
        min_val = times.min()
        max_val = times.max()
        ax.set_ylim(min_val * 0.7, max_val * 1.8)

        # Add actual time labels on top of bars
        for i, (method, time) in enumerate(zip(methods, times)):
            ax.text(i, time * 1.15, f'{time:.2f}s',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Create exactly 6 tick positions for log scale
        y_ticks = np.logspace(np.log10(min_val), np.log10(max_val), 6)
        y_labels = [f'{tick:.2f}' for tick in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=13)

        # Remove minor ticks to avoid cluttered grid
        ax.minorticks_off()

        # Formatting
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        ax.set_title(f'Generation Time Comparison - {dataset_display}',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Methods (ordered by time)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Generation Time (seconds, log scale)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=13)

        # Grid only on major ticks, cleaner appearance
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.0, which='major')
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_dir / f"generation_time_{dataset}.pdf"
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"   Generated: {output_file}")
        plt.close()


def plot_training_time(df, output_dir):
    """Create separate bar charts for training time comparison using relative values."""
    # Group by method and dataset, average across classifiers
    grouped = df.groupby(['Method', 'Dataset'])['Training_Time'].mean().reset_index()
    datasets = sorted(grouped['Dataset'].unique())

    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = grouped[grouped['Dataset'] == dataset].copy()

        # Sort by training time (ascending order - smallest to largest)
        dataset_df = dataset_df.sort_values('Training_Time')

        # Get methods and their times
        methods = dataset_df['Method'].values
        times = dataset_df['Training_Time'].values

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        x = np.arange(len(methods))
        bars = ax.bar(x, times, width=0.6, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Color bars with consistent colors for each method
        for bar, method in zip(bars, methods):
            bar.set_facecolor(METHOD_COLORS.get(method, '#95a5a6'))

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Set y-axis limits to accommodate labels (with extra space on top)
        min_val = times.min()
        max_val = times.max()
        ax.set_ylim(min_val * 0.7, max_val * 1.8)

        # Add actual time labels on top of bars
        for i, (method, time) in enumerate(zip(methods, times)):
            ax.text(i, time * 1.15, f'{time:.1f}s',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Create exactly 6 tick positions for log scale
        y_ticks = np.logspace(np.log10(min_val), np.log10(max_val), 6)
        y_labels = [f'{tick:.1f}' for tick in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=13)

        # Remove minor ticks to avoid cluttered grid
        ax.minorticks_off()

        # Formatting
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        ax.set_title(f'Training Time Comparison - {dataset_display}',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Methods (ordered by time)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Time (seconds, log scale)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=13)

        # Grid only on major ticks, cleaner appearance
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.0, which='major')
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_dir / f"training_time_{dataset}.pdf"
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"   Generated: {output_file}")
        plt.close()


def plot_testing_time(df, output_dir):
    """Create separate bar charts for testing time comparison using relative values."""
    # Group by method and dataset, average across classifiers
    grouped = df.groupby(['Method', 'Dataset'])['Testing_Time'].mean().reset_index()
    datasets = sorted(grouped['Dataset'].unique())

    for dataset in datasets:
        # Filter data for this dataset
        dataset_df = grouped[grouped['Dataset'] == dataset].copy()

        # Sort by testing time (ascending order - smallest to largest)
        dataset_df = dataset_df.sort_values('Testing_Time')

        # Get methods and their times
        methods = dataset_df['Method'].values
        times = dataset_df['Testing_Time'].values

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        x = np.arange(len(methods))
        bars = ax.bar(x, times, width=0.6, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Color bars with consistent colors for each method
        for bar, method in zip(bars, methods):
            bar.set_facecolor(METHOD_COLORS.get(method, '#95a5a6'))

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Set y-axis limits to accommodate labels (with extra space on top)
        min_val = times.min()
        max_val = times.max()
        ax.set_ylim(min_val * 0.7, max_val * 1.8)

        # Add actual time labels on top of bars
        for i, (method, time) in enumerate(zip(methods, times)):
            ax.text(i, time * 1.15, f'{time:.2f}s',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Create exactly 6 tick positions for log scale
        y_ticks = np.logspace(np.log10(min_val), np.log10(max_val), 6)
        y_labels = [f'{tick:.2f}' for tick in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=13)

        # Remove minor ticks to avoid cluttered grid
        ax.minorticks_off()

        # Formatting
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        ax.set_title(f'Testing Time Comparison - {dataset_display}',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Methods (ordered by time)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Testing Time (seconds, log scale)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=13)

        # Grid only on major ticks, cleaner appearance
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.0, which='major')
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_dir / f"testing_time_{dataset}.pdf"
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"   Generated: {output_file}")
        plt.close()


def main():
    """Main function to generate all comparison figures."""
    print("=" * 70)
    print("Generating Comparison Figures")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect data
    print("\n[1/4] Collecting generation time data...")
    gen_time_df = collect_generation_times()
    print(f"   Found {len(gen_time_df)} generation time records")

    print("\n[2/4] Collecting training and testing time data...")
    train_df, test_df = collect_training_testing_times()
    print(f"   Found {len(train_df)} training time records")
    print(f"   Found {len(test_df)} testing time records")

    # Generate figures
    print("\n[3/4] Generating figures...")

    print("\n   Creating generation time figures (one per dataset)...")
    plot_generation_time(gen_time_df, OUTPUT_DIR)

    print("\n   Creating training time figures (one per dataset)...")
    plot_training_time(train_df, OUTPUT_DIR)

    print("\n   Creating testing time figures (one per dataset)...")
    plot_testing_time(test_df, OUTPUT_DIR)

    print("\n[4/4] All figures generated successfully!")
    print("\n" + "=" * 70)
    print("Output location:", OUTPUT_DIR)
    print("=" * 70)
    print("\nGenerated PDF files (12 total - 3 time types × 4 datasets):")

    print("\n  Generation Time PDFs:")
    for dataset in DATASETS:
        print(f"    - generation_time_{dataset}.pdf")

    print("\n  Training Time PDFs:")
    for dataset in DATASETS:
        print(f"    - training_time_{dataset}.pdf")

    print("\n  Testing Time PDFs:")
    for dataset in DATASETS:
        print(f"    - testing_time_{dataset}.pdf")
    print()


if __name__ == "__main__":
    main()
