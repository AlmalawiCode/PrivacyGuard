# Python Scripts

Python scripts for generating manuscript figures and analysis.

## Scripts

### 1. plot_complexity.py
Generates empirical complexity analysis plots from the Java ComplexityBenchmark results.

**Features:**
- Fits complexity models (O(n), O(n²), O(n log n), O(log n)) to measured data
- Generates individual method plots with curve fitting
- Creates comparison plots across all methods
- Produces log-log scale and normalized efficiency plots
- Outputs analysis report with R² scores

**Usage:**
```bash
# First run the complexity benchmark from the Java menu (option 8)
# Then generate plots:
python scripts/plot_complexity.py
```

**Output:**
- `output/complexity_plots/complexity_<method>.pdf` - Individual method plots
- `output/complexity_plots/complexity_comparison.pdf` - All methods comparison
- `output/complexity_plots/complexity_loglog.pdf` - Log-log scale plot
- `output/complexity_plots/complexity_normalized.pdf` - Normalized efficiency
- `output/complexity_plots/complexity_analysis_report.txt` - Text report
- `output/complexity_plots/size_vs_time.csv` - Data size vs time CSV

### 2. generate_dataset_comparison_figures.py
Generates training and testing time comparison figures for each dataset, comparing across methods and classifiers.

### 3. generate_comparison_figures.py
Generates three comparison figures:
- Total Generation Time comparison
- Training Time comparison
- Testing Time comparison

### 4. collect_all_times.py
Collects all timing data (training, testing, generation times) from evaluation results into a summary CSV file.

## Requirements

```bash
pip install matplotlib pandas numpy scipy
```

## Usage

Run from the project root directory:

```bash
cd PrivacyGuard

# Complexity analysis (after running menu option 8)
python scripts/plot_complexity.py

# Time comparison figures
python scripts/collect_all_times.py
python scripts/generate_comparison_figures.py
python scripts/generate_dataset_comparison_figures.py
```

## Output

Figures are saved as PDF files in the `output/` directory or specified output directory.

## Author
Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
