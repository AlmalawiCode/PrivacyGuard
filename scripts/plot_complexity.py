#!/usr/bin/env python3
"""
Empirical Complexity Analysis Plotter

This script reads the benchmark results from the Java ComplexityBenchmark class
and creates visualizations to analyze the computational complexity of each
synthetic data generation method.

It fits different complexity models (O(n), O(n²), O(n log n)) to the data
and determines which model best describes each method's behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import os

# Input/output paths - will search for any complexity_benchmark_*.csv file
RESULTS_DIR = "output"
OUTPUT_DIR = "output/complexity_plots"

def find_results_file():
    """Find the most recent complexity benchmark CSV file."""
    import glob
    files = glob.glob(os.path.join(RESULTS_DIR, "complexity_benchmark_*.csv"))
    if not files:
        return None
    # Return the most recently modified file
    return max(files, key=os.path.getmtime)

# Complexity model functions
def linear(n, a, b):
    """O(n) - Linear complexity"""
    return a * n + b

def quadratic(n, a, b, c):
    """O(n²) - Quadratic complexity"""
    return a * n**2 + b * n + c

def nlogn(n, a, b):
    """O(n log n) - Linearithmic complexity"""
    return a * n * np.log(n + 1) + b

def logarithmic(n, a, b):
    """O(log n) - Logarithmic complexity"""
    return a * np.log(n + 1) + b

def cubic(n, a, b, c, d):
    """O(n³) - Cubic complexity"""
    return a * n**3 + b * n**2 + c * n + d

# Model definitions
MODELS = {
    'O(n)': (linear, ['a', 'b']),
    'O(n²)': (quadratic, ['a', 'b', 'c']),
    'O(n log n)': (nlogn, ['a', 'b']),
    'O(log n)': (logarithmic, ['a', 'b']),
    # 'O(n³)': (cubic, ['a', 'b', 'c', 'd']),  # Uncomment if needed
}

# Colors for methods
METHOD_COLORS = {
    'PrivacyGuard': '#2ecc71',       # Green - proposed method
    'Equal_Width_Binning': '#3498db', # Blue
    'kmeans': '#e74c3c',              # Red
    'k_anonymity': '#9b59b6',         # Purple
    'LaplaceDP': '#f39c12',           # Orange
}


def load_data(csv_path):
    """Load and preprocess the benchmark results."""
    df = pd.read_csv(csv_path)

    # Calculate average time per method and percentage
    avg_df = df.groupby(['method', 'percentage', 'num_instances']).agg({
        'time_ms': ['mean', 'std', 'min', 'max']
    }).reset_index()

    avg_df.columns = ['method', 'percentage', 'num_instances',
                      'avg_time_ms', 'std_time_ms', 'min_time_ms', 'max_time_ms']

    return df, avg_df


def fit_complexity_models(n_values, time_values):
    """
    Fit different complexity models to the data and return the best fit.
    Returns a dictionary with fit results for each model.
    """
    results = {}

    for model_name, (model_func, param_names) in MODELS.items():
        try:
            # Initial parameter guesses
            p0 = [1.0] * len(param_names)

            # Fit the model
            popt, pcov = curve_fit(model_func, n_values, time_values,
                                   p0=p0, maxfev=10000)

            # Calculate predictions and R² score
            predictions = model_func(n_values, *popt)
            ss_res = np.sum((time_values - predictions) ** 2)
            ss_tot = np.sum((time_values - np.mean(time_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Calculate RMSE
            rmse = np.sqrt(np.mean((time_values - predictions) ** 2))

            results[model_name] = {
                'params': dict(zip(param_names, popt)),
                'r_squared': r_squared,
                'rmse': rmse,
                'predictions': predictions,
                'model_func': model_func,
                'popt': popt
            }
        except Exception as e:
            print(f"  Warning: Could not fit {model_name}: {e}")
            results[model_name] = None

    return results


def determine_best_model(fit_results):
    """Determine the best fitting model based on R² score."""
    best_model = None
    best_r2 = -np.inf

    for model_name, result in fit_results.items():
        if result is not None and result['r_squared'] > best_r2:
            best_r2 = result['r_squared']
            best_model = model_name

    return best_model, best_r2


def plot_individual_methods(avg_df, output_dir):
    """Create individual plots for each method - one PDF per method."""
    methods = avg_df['method'].unique()
    analysis_results = {}

    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 7))
        method_data = avg_df[avg_df['method'] == method].sort_values('num_instances')

        n_values = method_data['num_instances'].values
        time_values = method_data['avg_time_ms'].values / 1000.0  # Convert to seconds
        std_values = method_data['std_time_ms'].values / 1000.0 if method_data['std_time_ms'].values is not None else None

        # Plot actual data points
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(n_values, time_values, 'o-', color=color, markersize=10,
                linewidth=2.5, label='Measured')

        # Fit complexity models (still use ms for fitting, convert for display)
        fit_results = fit_complexity_models(n_values, method_data['avg_time_ms'].values)
        best_model, best_r2 = determine_best_model(fit_results)

        analysis_results[method] = {
            'fit_results': fit_results,
            'best_model': best_model,
            'best_r2': best_r2
        }

        # Plot fitted curves
        n_smooth = np.linspace(n_values.min(), n_values.max(), 100)

        for model_name, result in fit_results.items():
            if result is not None:
                predictions = result['model_func'](n_smooth, *result['popt']) / 1000.0  # Convert to seconds
                linestyle = '-' if model_name == best_model else '--'
                alpha = 1.0 if model_name == best_model else 0.4
                linewidth = 2.5 if model_name == best_model else 1.5
                ax.plot(n_smooth, predictions, linestyle=linestyle, alpha=alpha,
                        linewidth=linewidth, label=f"{model_name} (R²={result['r_squared']:.3f})")

        ax.set_xlabel('Data Size (n)', fontsize=14)
        ax.set_ylabel('Time (seconds)', fontsize=14)
        ax.set_title(f'{method}\nBest fit: {best_model} (R²={best_r2:.3f})', fontsize=16)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # Format x-axis with K notation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'complexity_{method}.pdf'), bbox_inches='tight')
        plt.close()
        print(f"    Saved: complexity_{method}.pdf")

    return analysis_results


def plot_comparison(avg_df, output_dir):
    """Create a comparison plot of all methods."""
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = avg_df['method'].unique()

    for method in methods:
        method_data = avg_df[avg_df['method'] == method].sort_values('num_instances')

        n_values = method_data['num_instances'].values
        time_values = method_data['avg_time_ms'].values / 1000.0  # Convert to seconds

        color = METHOD_COLORS.get(method, '#333333')
        linewidth = 3 if method == 'PrivacyGuard' else 2

        ax.plot(n_values, time_values, 'o-', color=color, markersize=8,
                label=method, linewidth=linewidth)

    ax.set_xlabel('Data Size (n)', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Computational Complexity Comparison\nAll Synthetic Data Generation Methods', fontsize=16)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_log_scale(avg_df, output_dir):
    """Create a log-log plot to better visualize complexity differences."""
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = avg_df['method'].unique()

    for method in methods:
        method_data = avg_df[avg_df['method'] == method].sort_values('num_instances')

        n_values = method_data['num_instances'].values
        time_values = method_data['avg_time_ms'].values / 1000.0  # Convert to seconds

        color = METHOD_COLORS.get(method, '#333333')
        linewidth = 3 if method == 'PrivacyGuard' else 2

        ax.plot(n_values, time_values, 'o-', color=color,
                markersize=8, label=method, linewidth=linewidth)

    ax.set_xlabel('Data Size (n)', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Computational Complexity (Log-Log Scale)', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_loglog.pdf'), bbox_inches='tight')
    plt.close()


def plot_normalized(avg_df, output_dir):
    """Create a normalized plot (time per instance) to compare efficiency."""
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = avg_df['method'].unique()

    for method in methods:
        method_data = avg_df[avg_df['method'] == method].sort_values('num_instances')

        n_values = method_data['num_instances'].values
        time_values = method_data['avg_time_ms'].values / 1000.0  # Convert to seconds

        # Normalize: time per 1000 instances (in seconds)
        normalized_time = (time_values / n_values) * 1000

        color = METHOD_COLORS.get(method, '#333333')
        linewidth = 3 if method == 'PrivacyGuard' else 2

        ax.plot(n_values, normalized_time, 'o-', color=color,
                markersize=8, label=method, linewidth=linewidth)

    ax.set_xlabel('Data Size (n)', fontsize=14)
    ax.set_ylabel('Time per 1000 Instances (seconds)', fontsize=14)
    ax.set_title('Normalized Execution Time\n(Lower is better for linear complexity)', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_normalized.pdf'), bbox_inches='tight')
    plt.close()


def generate_summary_report(analysis_results, avg_df, output_dir):
    """Generate a text summary of the complexity analysis."""
    report_path = os.path.join(output_dir, 'complexity_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("     EMPIRICAL COMPLEXITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # DATA SIZE vs TIME TABLE (most important!)
        f.write("=" * 80 + "\n")
        f.write("  DATA SIZE (n) vs EXECUTION TIME (ms) - RAW DATA\n")
        f.write("=" * 80 + "\n\n")

        methods = avg_df['method'].unique()
        sizes = sorted(avg_df['num_instances'].unique())

        # Header
        f.write(f"{'n (instances)':<15}")
        for method in methods:
            f.write(f"{method:<20}")
        f.write("\n")
        f.write("-" * (15 + 20 * len(methods)) + "\n")

        # Data rows
        for size in sizes:
            f.write(f"{size:<15,}")
            for method in methods:
                row = avg_df[(avg_df['method'] == method) & (avg_df['num_instances'] == size)]
                if not row.empty:
                    time_ms = row['avg_time_ms'].values[0]
                    f.write(f"{time_ms:<20,.0f}")
                else:
                    f.write(f"{'N/A':<20}")
            f.write("\n")

        f.write("\n\n")

        # SUMMARY OF BEST FIT MODELS
        f.write("=" * 80 + "\n")
        f.write("  COMPLEXITY ANALYSIS - BEST FIT MODELS\n")
        f.write("=" * 80 + "\n\n")

        for method, results in analysis_results.items():
            f.write(f"Method: {method}\n")
            f.write(f"  Best fit model: {results['best_model']}\n")
            f.write(f"  R² score: {results['best_r2']:.4f}\n")

            if results['fit_results'].get(results['best_model']):
                params = results['fit_results'][results['best_model']]['params']
                f.write(f"  Parameters: {params}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  DETAILED MODEL COMPARISON (R² scores)\n")
        f.write("=" * 80 + "\n\n")

        for method, results in analysis_results.items():
            f.write(f"\n{method}:\n")
            f.write("-" * 40 + "\n")

            for model_name, fit_result in results['fit_results'].items():
                if fit_result is not None:
                    f.write(f"  {model_name}:\n")
                    f.write(f"    R² = {fit_result['r_squared']:.4f}\n")
                    f.write(f"    RMSE = {fit_result['rmse']:.2f} ms\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("  INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("""
O(n)      - Linear: Time grows proportionally with data size
            Ideal for scalability

O(n log n)- Linearithmic: Slightly worse than linear
            Still excellent scalability

O(n²)     - Quadratic: Time grows with square of data size
            May be problematic for large datasets

O(log n)  - Logarithmic: Almost constant time
            Extremely efficient

R² Score Interpretation:
  > 0.95: Excellent fit
  > 0.85: Good fit
  > 0.70: Moderate fit
  < 0.70: Poor fit (may need different model)
""")

    print(f"\nReport saved to: {report_path}")
    return report_path


def save_size_time_csv(avg_df, output_dir):
    """Save a clean CSV with just data size and time for each method."""
    csv_path = os.path.join(output_dir, 'size_vs_time.csv')

    # Pivot the data: rows = sizes, columns = methods
    pivot_df = avg_df.pivot_table(
        index='num_instances',
        columns='method',
        values='avg_time_ms'
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={'num_instances': 'n_instances'})

    pivot_df.to_csv(csv_path, index=False)
    print(f"Size vs Time CSV saved to: {csv_path}")

    return csv_path


def main():
    print("=" * 50)
    print("  Complexity Analysis Plotter")
    print("=" * 50 + "\n")

    # Find results file
    results_csv = find_results_file()
    if results_csv is None:
        print(f"Error: No complexity benchmark CSV files found in {RESULTS_DIR}/")
        print("Please run the Java ComplexityBenchmark first (Menu option 7).")
        return

    print(f"Found results file: {results_csv}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading benchmark results...")
    df, avg_df = load_data(results_csv)
    print(f"Loaded {len(df)} measurements for {df['method'].nunique()} methods")

    # Print data sizes prominently
    sizes = sorted(df['num_instances'].unique())
    print(f"\nData sizes (n) tested:")
    for size in sizes:
        print(f"  n = {size:,}")

    # Generate plots
    print("\nGenerating plots...")

    print("  - Individual method plots with curve fitting...")
    analysis_results = plot_individual_methods(avg_df, OUTPUT_DIR)

    print("  - Comparison plot...")
    plot_comparison(avg_df, OUTPUT_DIR)

    print("  - Log-log scale plot...")
    plot_log_scale(avg_df, OUTPUT_DIR)

    print("  - Normalized efficiency plot...")
    plot_normalized(avg_df, OUTPUT_DIR)

    # Save size vs time CSV
    print("\nSaving size vs time data...")
    save_size_time_csv(avg_df, OUTPUT_DIR)

    # Generate summary report
    print("\nGenerating analysis report...")
    generate_summary_report(analysis_results, avg_df, OUTPUT_DIR)

    # Print DATA SIZE vs TIME table to console (most important!)
    print("\n" + "=" * 80)
    print("  DATA SIZE (n) vs EXECUTION TIME (ms)")
    print("=" * 80)

    methods = list(avg_df['method'].unique())
    header = f"{'n':<12}"
    for method in methods:
        header += f"{method:<16}"
    print(header)
    print("-" * len(header))

    for size in sizes:
        row = f"{size:<12,}"
        for method in methods:
            data = avg_df[(avg_df['method'] == method) & (avg_df['num_instances'] == size)]
            if not data.empty:
                time_ms = data['avg_time_ms'].values[0]
                row += f"{time_ms:<16,.0f}"
            else:
                row += f"{'N/A':<16}"
        print(row)

    # Print complexity summary
    print("\n" + "=" * 50)
    print("  COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 50 + "\n")

    for method, results in analysis_results.items():
        complexity = results['best_model']
        r2 = results['best_r2']
        print(f"  {method:25s} -> {complexity:12s} (R² = {r2:.3f})")

    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    for method in methods:
        print(f"  - complexity_{method}.pdf")
    print("  - complexity_comparison.pdf")
    print("  - complexity_loglog.pdf")
    print("  - complexity_normalized.pdf")
    print("  - complexity_analysis_report.txt")
    print("  - size_vs_time.csv")


if __name__ == "__main__":
    main()
