# Figure Generation Scripts

Python scripts for generating manuscript figures.

## Scripts

### 1. generate_dataset_comparison_figures.py
Generates training and testing time comparison figures for each dataset, comparing across methods and classifiers.

### 2. generate_comparison_figures.py
Generates three comparison figures:
- Total Generation Time comparison
- Training Time comparison
- Testing Time comparison

### 3. collect_all_times.py
Collects all timing data (training, testing, generation times) from evaluation results into a summary CSV file.

## Requirements

```bash
pip install matplotlib pandas numpy
```

## Usage

Run from the project root directory:

```bash
cd PrivacyGuard
python scripts/collect_all_times.py
python scripts/generate_comparison_figures.py
python scripts/generate_dataset_comparison_figures.py
```

**Note:** Update the paths in each script to match your project directory before running.

## Output

Figures are saved as PDF files in the scripts directory or specified output directory.

## Author
Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
