#!/usr/bin/env python3
"""
Script to generate comparison figures from all_training_testing_times_summary.csv
Shows training and testing times for each dataset, comparing across methods and classifiers.
No averaging - each classifier shown separately.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths
BASE_DIR = Path("/home/abdul/claudeProjects/VWC-SyntheticData")
OUTPUT_DIR = Path(__file__).parent  # Save in same directory as script
SUMMARY_FILE = BASE_DIR / "output" / "figures" / "time_collection" / "all_training_testing_times_summary.csv"

# ============================================================================
# CONFIGURATION SECTION - Adjust these settings to customize the figures
# ============================================================================

# Classifier colors (modify these hex codes to change colors)
CLASSIFIER_COLORS = {
    "J48": "#e74c3c",           # Red
    "NaiveBayes": "#f39c12",    # Orange
    "RandomForest": "#3498db"   # Blue
}

# Figure size (width, height) in inches
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 5

# Bar properties
BAR_WIDTH = 0.25           # Width of each bar (0.1 to 0.4 recommended)
BAR_ALPHA = 0.8            # Transparency (0.0 to 1.0, where 1.0 is opaque)
BAR_EDGE_COLOR = "black"   # Color of bar edges
BAR_EDGE_WIDTH = 0.5       # Width of bar edges

# Font sizes
TITLE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 11
TICK_LABEL_FONT_SIZE = 10
VALUE_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 9

# Y-axis limits multipliers (for log scale)
Y_MIN_MULTIPLIER = 0.7     # Multiply minimum value by this (e.g., 0.7 = 70% of min)
Y_MAX_MULTIPLIER = 2.5     # Multiply maximum value by this (e.g., 2.5 = 250% of max)

# Value label position
VALUE_LABEL_MULTIPLIER =1.02  # Position above bar (1.15 = 15% above bar top)

# Legend properties
LEGEND_LOCATION = "upper center"  # Options: "upper left", "upper right", "upper center", etc.
LEGEND_COLUMNS = 3                # Number of columns in legend
LEGEND_FRAME_ALPHA = 0.95         # Legend background transparency
LEGEND_EDGE_COLOR = "black"       # Legend box edge color
LEGEND_EDGE_WIDTH = 0.0        # Legend box border thickness (try 0.5-3.0)

# Figure border (spine) properties
FIGURE_BORDER_COLOR = "black"     # Color of figure border
FIGURE_BORDER_WIDTH = 0.01         # Thickness of figure border (try 0.5-3.0)

# Grid properties
GRID_ALPHA = 0.3              # Grid line transparency
GRID_LINESTYLE = "--"         # Grid line style: "--", "-", "-.", ":"
GRID_LINEWIDTH = 1.0          # Grid line width

# Output format
OUTPUT_DPI = 300              # Resolution for PDF output (300 is publication quality)

# Number format for value labels
TRAINING_TIME_FORMAT = ".1f"  # Format for training time (e.g., ".1f" = 1 decimal place)
TESTING_TIME_FORMAT = ".2f"   # Format for testing time (e.g., ".2f" = 2 decimal places)

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


# Dataset name mapping (code name -> paper name)
DATASET_NAME_MAPPING = {
    'bot_loT': 'N-BaIoT',
    'Bot-IoT': 'N-BaIoT',
    'Edge-IIoTset': 'Edge-IIoTset',
    'CICIoT2023': 'CICIoT2023',
    'MQTTset': 'MQTTset'
}

def get_display_name(dataset):
    """Get the display name for a dataset (for paper consistency)."""
    return DATASET_NAME_MAPPING.get(dataset, dataset)

def load_data():
    """Load the summary CSV file."""
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found!")
        print("Please run collect_all_times.py first.")
        return None

    return pd.read_csv(SUMMARY_FILE)


def plot_dataset_comparison(df, dataset, time_type, output_dir):
    """
    Create comparison figure for a specific dataset and time type showing all methods and classifiers.
    time_type: 'Training' or 'Testing'
    """
    # Filter data for this dataset
    dataset_df = df[df['Dataset'] == dataset].copy()

    if dataset_df.empty:
        print(f"   No data found for {dataset}")
        return

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    methods = sorted(dataset_df['Method'].unique())
    classifiers = sorted(dataset_df['Classifier'].unique())

    # Prepare data for grouped bars
    n_methods = len(methods)
    n_classifiers = len(classifiers)
    x = np.arange(n_methods)
    width = BAR_WIDTH

    # Determine which time column to use
    time_column = f'{time_type}_Time_Seconds'

    # Plot bars for each classifier
    for idx, classifier in enumerate(classifiers):
        times = []
        for method in methods:
            method_classifier_data = dataset_df[(dataset_df['Method'] == method) &
                                                (dataset_df['Classifier'] == classifier)]
            if len(method_classifier_data) > 0:
                times.append(method_classifier_data[time_column].values[0])
            else:
                times.append(0)

        offset = width * (idx - 1)
        bars = ax.bar(x + offset, times, width,
                       color=CLASSIFIER_COLORS.get(classifier, '#95a5a6'),
                       alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            if time > 0:
                format_str = TRAINING_TIME_FORMAT if time_type == 'Training' else TESTING_TIME_FORMAT
                ax.text(bar.get_x() + bar.get_width()/2, time * VALUE_LABEL_MULTIPLIER,
                        f'{time:{format_str}}', ha='center', va='bottom',
                        fontsize=VALUE_LABEL_FONT_SIZE, fontweight='bold')

    # Create simple legend with just 3 classifiers
    from matplotlib.patches import Patch

    legend_elements = []
    for classifier in classifiers:
        legend_elements.append(Patch(facecolor=CLASSIFIER_COLORS.get(classifier, '#95a5a6'),
                                    edgecolor='black', label=classifier, alpha=0.8))

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set y-axis limits with extra space for legend at top
    all_times = dataset_df[time_column].values
    min_time = all_times.min()
    max_time = all_times.max()
    ax.set_ylim(min_time * Y_MIN_MULTIPLIER, max_time * Y_MAX_MULTIPLIER)

    ax.set_ylabel(f'{time_type} Time (seconds, log scale)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('Method', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    display_name = get_display_name(dataset)
    #ax.set_title(f'{display_name} Dataset', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)

    # Add simple legend with just classifiers
    legend = ax.legend(handles=legend_elements, fontsize=LEGEND_FONT_SIZE, loc=LEGEND_LOCATION,
             framealpha=LEGEND_FRAME_ALPHA, ncol=LEGEND_COLUMNS, edgecolor=LEGEND_EDGE_COLOR, fancybox=False)
    # Set legend border width
    legend.get_frame().set_linewidth(LEGEND_EDGE_WIDTH)

    ax.grid(axis='y', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, which='major')
    ax.set_axisbelow(True)
    ax.minorticks_off()

    # Set figure border (spine) thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(FIGURE_BORDER_WIDTH)
        spine.set_color(FIGURE_BORDER_COLOR)

    plt.tight_layout()

    # Save figure
    dataset_safe = dataset.replace(' ', '_').replace('-', '_')
    output_file = output_dir / f"{time_type.lower()}_time_{dataset_safe}.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=OUTPUT_DPI)
    print(f"   Generated: {output_file}")
    plt.close()


def main():
    """Main function to generate all comparison figures."""
    print("=" * 70)
    print("Generating Dataset Comparison Figures")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data from summary CSV...")
    df = load_data()
    if df is None:
        return

    print(f"   Loaded {len(df)} records")

    # Get unique datasets
    datasets = sorted(df['Dataset'].unique())
    print(f"\n[2/3] Found {len(datasets)} datasets: {', '.join(datasets)}")

    # Generate figures for each dataset and time type
    print("\n[3/3] Generating comparison figures for each dataset...")
    time_types = ['Training', 'Testing']

    for dataset in datasets:
        print(f"\n   Processing: {dataset}")
        for time_type in time_types:
            print(f"      - {time_type} time")
            plot_dataset_comparison(df, dataset, time_type, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)
    print(f"\nOutput location: {OUTPUT_DIR}")
    print(f"\nGenerated {len(datasets) * 2} PDF files:")
    print("\nTraining Time PDFs:")
    for dataset in datasets:
        dataset_safe = dataset.replace(' ', '_').replace('-', '_')
        print(f"  - training_time_{dataset_safe}.pdf")
    print("\nTesting Time PDFs:")
    for dataset in datasets:
        dataset_safe = dataset.replace(' ', '_').replace('-', '_')
        print(f"  - testing_time_{dataset_safe}.pdf")
    print()


if __name__ == "__main__":
    main()
