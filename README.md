# PrivacyGuard

**Privacy-Preserving Synthetic Data Generation for IoT Sensor Networks**

[![Java](https://img.shields.io/badge/Java-11%2B-orange)](https://www.oracle.com/java/)
[![Apache Ant](https://img.shields.io/badge/Build-Apache%20Ant-blue)](https://ant.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Author: Abdulmohsen Almalawi <balmalowy@kau.edu.sa>

---

## Overview

PrivacyGuard is a clustering-based approach for privacy-preserving data publishing in IoT sensor networks. It uses Various-Width Clustering (VWC) to transform original data into privacy-preserving synthetic representations while maintaining utility for machine learning tasks.

The approach achieves O(mnc) time complexity, where m is the number of features, n is the number of records, and c is the number of clusters per feature. This compares favourably to k-means clustering which requires O(mnkI) for k clusters over I iterations.

## Features

- **PrivacyGuard (VWC)**: Various-Width Clustering for privacy-preserving transformation
- **Baseline Methods**: Equal Width Binning, K-Means, K-Anonymity, Laplace DP
- **Classification Evaluation**: J48, Random Forest, Naive Bayes
- **Privacy Attack Evaluation**: Re-Identification Attack, Linkage Attack
- **Parameter Exploration**: Analyze impact of s_max on utility/privacy trade-off
- **Complexity Benchmark**: Empirical computational complexity analysis

---

## Prerequisites

- **Java JDK 11** or higher
- **Apache Ant** (for building)

### Check Java Version
```bash
java -version
```

### Check Ant Version
```bash
ant -version
```

---

## Quick Start

### Step 1: Build the Project
```bash
cd PrivacyGuard
ant compile
```

### Step 2: Run the Application
```bash
ant run
```

### Step 3: Follow the Menu
The menu-driven interface will guide you through all operations.

---

## Main Menu Options

```
[1] Generate Synthetic Data - Create privacy-preserving datasets
[2] Generate Train/Test Indices - Create consistent train/test splits
[3] Classification Evaluation - Evaluate classifiers on all methods
[4] Generate LaTeX Tables - Convert results to LaTeX format
[5] Privacy Attack Evaluation - Test re-identification & linkage attacks
[6] Parameter Exploration - Analyze impact of s_max on utility/privacy
[7] Explore Datasets - View dataset details and distributions
[8] Complexity Benchmark - Measure computational complexity empirically
[0] Exit
```

---

## Recommended Workflow

### 1. Generate Synthetic Data
```
Select option: 1
Select dataset: (choose dataset)
Select method: All methods
```
Generates synthetic data using all methods:
- PrivacyGuard (VWC) with s_max=5
- Equal Width Binning (10 bins)
- K-Means (k=1000)
- K-Anonymity (k=5)
- Laplace DP (epsilon=1.0)

### 2. Generate Train/Test Indices
```
Select option: 2
Select: All datasets
```
Creates stratified 70/30 train/test splits for reproducibility.

### 3. Classification Evaluation
```
Select option: 3
Select dataset: (choose dataset)
```
Trains J48, Random Forest, and Naive Bayes on synthetic data and compares with original.

### 4. Generate LaTeX Tables
```
Select option: 4
```
Converts evaluation results to LaTeX tables for publication.

### 5. Privacy Attack Evaluation
```
Select option: 5
Select dataset: (choose dataset)
```
Runs Re-Identification Attack and Linkage Attack.

### 6. Parameter Exploration (Optional)
```
Select option: 6
Select dataset: (choose dataset)
```
Analyzes impact of s_max values {5, 10, 15, 20, 25, 30, 35}.

### 7. Complexity Benchmark
```
Select option: 8
Select dataset: Edge-IIoTset (recommended)
```
Measures computational complexity empirically by testing all methods at 10%-100% data sizes.
After running, generate plots with:
```bash
python scripts/plot_complexity.py
```

---

## Project Structure

```
PrivacyGuard/
├── README.md                 # This file
├── build.xml                 # Ant build file
├── config.properties         # Configuration parameters
├── run.sh                    # Linux/Mac run script
├── run.bat                   # Windows run script
├── .gitignore               # Git ignore file
├── src/                      # Java source files
│   ├── MenuDrivenInterface.java
│   ├── SyntheticDataGenerator.java
│   ├── IndicesGenerator.java
│   ├── ClassificationEvaluator.java
│   ├── LaTeXTableGenerator.java
│   ├── PrivacyAttackEvaluator.java
│   ├── SyntheticDataGenerationWithParameterExploration.java
│   ├── ComplexityBenchmark.java
│   ├── PrivacyGuardGenerator.java
│   ├── EqualWidthBinning.java
│   ├── KMeansBaseline.java
│   ├── KAnonymity.java
│   ├── LaplaceDPBaseline.java
│   ├── ReIdentificationAttack.java
│   ├── LinkageAttack.java
│   └── ConfigLoader.java
├── lib/                      # Required JAR files
│   ├── weka.jar
│   └── WekaKNNVWC_1.1.jar
├── datasets/                 # Dataset files
│   └── Original/             # ARFF files
│       ├── bot_loT.arff
│       ├── CICIoT2023.arff
│       ├── Edge-IIoTset.arff
│       └── MQTTset.arff
├── scripts/                  # Python scripts for figures
│   ├── README.md
│   ├── plot_complexity.py
│   ├── generate_comparison_figures.py
│   ├── generate_dataset_comparison_figures.py
│   └── collect_all_times.py
└── output/                   # Generated results (organized by menu option)
    ├── 1_synthetic_data/     # [Menu 1] Synthetic datasets
    ├── 2_indices/            # [Menu 2] Train/test indices
    ├── 3_classification/     # [Menu 3] Classification results
    ├── 4_latex/              # [Menu 4] LaTeX tables
    ├── 5_privacy_attacks/    # [Menu 5] Privacy attack results
    ├── 6_parameter_exploration/  # [Menu 6] Parameter analysis
    └── complexity_plots/     # [Menu 8] Complexity benchmark plots
```

---

## Datasets

**Note:** All datasets are provided as compressed tar.xz files to reduce repository size. Before running the application, extract the files in the `datasets/Original/` folder.

```bash
cd datasets/Original
tar -xf "*.tar.xz"
```

| Dataset | Total Records | Normal | Abnormal | Attack Types | Features |
|---------|---------------|--------|----------|--------------|----------|
| N-BaIoT (bot_loT) | 146,609 | 61,565 | 85,044 | 8 | 115 |
| CICIoT2023 | 352,274 | 7,686 | 344,588 | 7 | 46 |
| Edge-IIoTset | 525,263 | 335,605 | 189,658 | 13 | 70 |
| MQTTset | 45,600 | 420 | 45,180 | 5 | 30 |

---

## Configuration

Parameters can be modified in `config.properties`:

```properties
# PrivacyGuard parameters
privacyguard.s_max=5
privacyguard.beta_percent=5

# Baseline parameters
ewb.num_bins=100
kmeans.k=1000
kanonymity.k=5
laplace.epsilon=1.0

# Evaluation parameters
evaluation.train_ratio=0.7
evaluation.num_runs=5
evaluation.random_seed=180
```

---

## Output Files

After running, results are saved in organized folders:

| Menu Option | Output Folder | Contents |
|-------------|---------------|----------|
| 1. Synthetic Data | `output/1_synthetic_data/` | Synthetic ARFF files by method |
| 2. Indices | `output/2_indices/` | Train/test index files |
| 3. Classification | `output/3_classification/` | CSV evaluation results |
| 4. LaTeX Tables | `output/4_latex/` | LaTeX table files |
| 5. Privacy Attacks | `output/5_privacy_attacks/` | Attack evaluation results |
| 6. Parameter Exploration | `output/6_parameter_exploration/` | Parameter analysis reports |
| 8. Complexity Benchmark | `output/complexity_plots/` | Complexity analysis plots |

---

## Troubleshooting

### Out of Memory Error
Increase Java heap size in `build.xml`:
```xml
<jvmarg value="-Xmx16g"/>
```

### Compilation Errors
Ensure Java 11+ is installed and JAVA_HOME is set correctly.

### Missing Dependencies
Verify `lib/` contains `weka.jar` and `WekaKNNVWC_1.1.jar`.

### Java Module Access Error (Java 9+)
The build.xml already includes the necessary JVM argument:
```xml
<jvmarg value="--add-opens=java.base/java.lang=ALL-UNNAMED"/>
```

---

## License

This project is provided for academic and research purposes.

## Contact

Abdulmohsen Almalawi
Email: balmalowy@kau.edu.sa
