#!/usr/bin/env python3
"""
Script to collect all training and testing times from classification results.
Creates a comprehensive summary file with all timing data.
"""

import pandas as pd
from pathlib import Path

# Define base paths
BASE_DIR = Path("/home/abdul/claudeProjects/VWC-SyntheticData")
RESULTS_DIR = BASE_DIR / "output" / "averaged_classification_results"
OUTPUT_DIR = Path(__file__).parent  # Save in same directory as script

# Define datasets
DATASETS = ["bot_loT", "CICIoT2023", "Edge-IIoTset", "MQTTset"]

# Dataset display names (code name -> paper name)
DATASET_DISPLAY_NAMES = {
    "bot_loT": "N-BaIoT",
    "Bot-IoT": "N-BaIoT",
    "CICIoT2023": "CICIoT2023",
    "Edge-IIoTset": "Edge-IIoTset",
    "MQTTset": "MQTTset"
}

# Method display names for better readability
METHOD_DISPLAY_NAMES = {
    "Equal_Width_Binning": "Equal Width Binning",
    "k_anonymity": "k-Anonymity",
    "kmeans": "K-Means",
    "LaplaceDP": "Laplace DP",
    "PrivacyGuard": "PrivacyGuard",
    "Original": "Original"
}


def collect_all_times():
    """Collect all training and testing times from averaged classification results."""
    all_data = []

    for dataset in DATASETS:
        result_file = RESULTS_DIR / f"averaged_results_{dataset}.csv"

        if result_file.exists():
            print(f"Processing {dataset}...")
            df = pd.read_csv(result_file)

            # Filter for training and testing times
            time_df = df[df['Metric'].isin(['Training Time (ns)', 'Testing Time (ns)'])]

            for _, row in time_df.iterrows():
                method = row['Dataset']
                classifier = row['Classifier']
                metric = row['Metric']
                time_ns = row['Value']
                time_sec = time_ns / 1e9  # Convert nanoseconds to seconds
                num_runs = row['Num_Runs']

                # Determine if it's training or testing
                time_type = 'Training' if 'Training' in metric else 'Testing'

                all_data.append({
                    'Dataset': DATASET_DISPLAY_NAMES.get(dataset, dataset),
                    'Method': METHOD_DISPLAY_NAMES.get(method, method),
                    'Classifier': classifier,
                    'Time_Type': time_type,
                    'Time_Seconds': time_sec,
                    'Time_Nanoseconds': time_ns,
                    'Num_Runs': num_runs
                })

    return pd.DataFrame(all_data)


def create_summary_statistics(df):
    """Create summary statistics for training and testing times including classifiers."""
    # Pivot table to organize data by dataset, method, classifier, and time type
    summary_df = df.pivot_table(
        index=['Dataset', 'Method', 'Classifier'],
        columns='Time_Type',
        values='Time_Seconds',
        aggfunc='mean'
    ).reset_index()

    # Rename columns for clarity
    if 'Training' in summary_df.columns and 'Testing' in summary_df.columns:
        summary_df = summary_df.rename(columns={
            'Training': 'Training_Time_Seconds',
            'Testing': 'Testing_Time_Seconds'
        })

    # Sort by dataset and method for better readability
    summary_df = summary_df.sort_values(['Dataset', 'Method', 'Classifier']).reset_index(drop=True)

    return summary_df


def main():
    """Main function to collect and save all timing data."""
    print("=" * 70)
    print("Collecting All Training and Testing Times")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all timing data
    print("\n[1/3] Collecting timing data from classification results...")
    all_times_df = collect_all_times()
    print(f"   Collected {len(all_times_df)} timing records")

    # Save detailed data
    detailed_file = OUTPUT_DIR / "all_training_testing_times_detailed.csv"
    all_times_df.to_csv(detailed_file, index=False)
    print(f"\n[2/3] Saved detailed timing data to:")
    print(f"   {detailed_file}")

    # Create and save summary statistics
    print("\n   Creating summary statistics...")
    summary_df = create_summary_statistics(all_times_df)
    summary_file = OUTPUT_DIR / "all_training_testing_times_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[3/3] Saved summary statistics to:")
    print(f"   {summary_file}")

    # Print some statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"\nTotal datasets: {all_times_df['Dataset'].nunique()}")
    print(f"Total methods: {all_times_df['Method'].nunique()}")
    print(f"Total classifiers: {all_times_df['Classifier'].nunique()}")
    print(f"Total records: {len(all_times_df)}")

    print("\n" + "=" * 70)
    print("Files Generated:")
    print("=" * 70)
    print(f"1. {detailed_file.name}")
    print(f"   - Contains all individual timing records")
    print(f"   - Columns: Dataset, Method, Classifier, Time_Type, Time_Seconds, etc.")
    print(f"\n2. {summary_file.name}")
    print(f"   - Contains training and testing times for each classifier")
    print(f"   - Columns: Dataset, Method, Classifier, Training_Time_Seconds, Testing_Time_Seconds")
    print()


if __name__ == "__main__":
    main()
