package privacyguard;

import java.io.*;
import java.util.*;

/**
 * LaTeX Table Generator
 * Converts CSV evaluation results to LaTeX tables (Option 1 format)
 * Method-centric: Each method is a row, metrics grouped by classifier
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class LaTeXTableGenerator {

    // ============================================================================
    // CONFIGURATION: Table Width Percentage
    // ============================================================================
    // Adjust this value to control the width of generated LaTeX tables
    // Examples: "1.0" for 100%, "0.90" for 90%, "0.85" for 85%, "0.80" for 80%
    // Set to "" (empty string) to disable resizing (table will use natural width)
    private static final String TABLE_WIDTH_PERCENTAGE = "1.0";
    // ============================================================================

    private static final String CLASSIFICATION_DIR = "output/3_classification/";
    private static final String PRIVACY_DIR = "output/5_privacy_attacks/";
    private static final String LATEX_OUTPUT_DIR = "output/4_latex/";

    private static final String[][] DATASETS = {
        {"Bot-IoT", "bot_loT"},
        {"CICIoT2023", "CICIoT2023"},
        {"MQTTset", "MQTTset"},
        {"Edge-IIoTset", "Edge-IIoTset"}
    };

    private static final String[] METHODS = {
        "Original",
        "PrivacyGuard",
        "Equal_Width_Binning",
        "KMeans",
        "KAnonymity",
        "LaplacianDP"
    };

    private static final String[] CLASSIFIERS = {"J48", "RandomForest", "NaiveBayes"};

    /**
     * Generate all LaTeX tables in one file (without document preamble)
     * Uses averaged results from multiple runs
     */
    public static void generateAllTablesInOneFile() throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║       Generating All LaTeX Tables in One File                ║");
        System.out.println("║       (Using Averaged Results from Multiple Runs)            ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Create output directory
        new File(LATEX_OUTPUT_DIR).mkdirs();

        // Output file
        String outputFile = LATEX_OUTPUT_DIR + "all_results_tables.tex";

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {

            // Generate table for each dataset
            for (String[] dataset : DATASETS) {
                String displayName = dataset[0];
                String datasetPrefix = dataset[1];

                // Input CSV file from classification results
                String inputFile = CLASSIFICATION_DIR + "evaluation_comparison_results_" + datasetPrefix + ".csv";

                if (!new File(inputFile).exists()) {
                    System.out.println("⊳ Skipping " + displayName + " (no averaged results file found)");
                    System.out.println("  Expected: " + inputFile);
                    continue;
                }

                System.out.println("Processing: " + displayName);

                // Parse CSV and extract metrics
                Map<String, Map<String, MetricValues>> results = parseCSV(inputFile);

                // Write table to file (without document preamble)
                writeLatexTable(writer, results, displayName, datasetPrefix);

                // Add spacing between tables
                writer.println();
                writer.println();
            }

            System.out.println();
            System.out.println("✓ All LaTeX tables generated successfully!");
            System.out.println("  Output: " + outputFile);
            System.out.println("  Source: Averaged results from multiple runs");
            System.out.println("  Note: No document preamble included - ready for inclusion");
            System.out.println();
        }
    }

    /**
     * Generate LaTeX table for a specific dataset (deprecated - kept for compatibility)
     * Uses averaged results from multiple runs
     */
    public static void generateLatexTable(String datasetPrefix) throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║  Generating LaTeX Table: " + padRight(datasetPrefix, 37) + "║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Input CSV file from classification results
        String inputFile = CLASSIFICATION_DIR + "evaluation_comparison_results_" + datasetPrefix + ".csv";

        if (!new File(inputFile).exists()) {
            System.out.println("✗ Error: Classification results file not found: " + inputFile);
            System.out.println("  Please run classification evaluation first (Menu option 3).");
            return;
        }

        // Parse CSV and extract metrics
        Map<String, Map<String, MetricValues>> results = parseCSV(inputFile);

        // Create output directory
        new File(LATEX_OUTPUT_DIR).mkdirs();

        // Generate LaTeX table (single file with preamble)
        String outputFile = LATEX_OUTPUT_DIR + datasetPrefix + "_results_table.tex";
        generateLatexFile(results, datasetPrefix, outputFile);

        System.out.println("✓ LaTeX table generated successfully!");
        System.out.println("  Output: " + outputFile);
        System.out.println("  Source: Averaged results from multiple runs");
        System.out.println();
    }

    /**
     * Generate summary ranking frequency table (Option 3 only)
     */
    public static void generateAllSummaryTables() throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║       Generating Summary Ranking Frequency Table             ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Create output directory
        new File(LATEX_OUTPUT_DIR).mkdirs();

        // Load all datasets
        Map<String, Map<String, Map<String, MetricValues>>> allDatasets = new HashMap<>();
        for (String[] dataset : DATASETS) {
            String displayName = dataset[0];
            String datasetPrefix = dataset[1];
            String inputFile = CLASSIFICATION_DIR + "evaluation_comparison_results_" + datasetPrefix + ".csv";

            if (new File(inputFile).exists()) {
                allDatasets.put(displayName, parseCSV(inputFile));
            }
        }

        // Generate Option 3: Ranking frequency table (excluding Original)
        String output3 = LATEX_OUTPUT_DIR + "summary_option3_ranking_frequency.tex";
        try (PrintWriter writer = new PrintWriter(new FileWriter(output3))) {
            generateOption3_RankingFrequency(writer, allDatasets);
        }
        System.out.println("✓ Generated: " + output3);

        System.out.println();
        System.out.println("✓ Summary ranking frequency table generated successfully!");
        System.out.println();
    }

    /**
     * Option 1: Average performance across all datasets
     */
    private static void generateOption1_AverageAcrossDatasets(PrintWriter writer,
            Map<String, Map<String, Map<String, MetricValues>>> allDatasets) {

        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Average Classification Results Across All Datasets}");
        writer.println("\\label{tab:summary_average}");
        writer.println("\\resizebox{\\textwidth}{!}{%");
        writer.println("\\begin{tabular}{l|ccc|ccc|ccc}");
        writer.println("\\toprule");
        writer.println("\\multirow{2}{*}{\\textbf{Method}} & " +
                      "\\multicolumn{3}{c|}{\\textbf{J48}} & " +
                      "\\multicolumn{3}{c|}{\\textbf{RandomForest}} & " +
                      "\\multicolumn{3}{c}{\\textbf{NaiveBayes}} \\\\");
        writer.println("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}");
        writer.println(" & M-Prec & M-F1 & M-Rec & " +
                      "M-Prec & M-F1 & M-Rec & " +
                      "M-Prec & M-F1 & M-Rec \\\\");
        writer.println("\\midrule");

        // Calculate averages across all datasets
        Map<String, Map<String, double[]>> methodAverages = new HashMap<>();

        for (String method : METHODS) {
            methodAverages.put(method, new HashMap<>());
            for (String classifier : CLASSIFIERS) {
                double[] sums = new double[3]; // precision, fmeasure, recall
                int count = 0;

                for (Map<String, Map<String, MetricValues>> datasetResults : allDatasets.values()) {
                    if (datasetResults.containsKey(method)) {
                        MetricValues mv = datasetResults.get(method).get(classifier);
                        sums[0] += calculateMacroAverage(mv.precisionValues);
                        sums[1] += calculateMacroAverage(mv.fmeasureValues);
                        sums[2] += calculateMacroAverage(mv.recallValues);
                        count++;
                    }
                }

                if (count > 0) {
                    double[] averages = {sums[0]/count, sums[1]/count, sums[2]/count};
                    methodAverages.get(method).put(classifier, averages);
                }
            }
        }

        // Write data rows
        for (String method : METHODS) {
            writer.print(formatMethodName(method));

            for (String classifier : CLASSIFIERS) {
                double[] avg = methodAverages.get(method).get(classifier);
                if (avg != null) {
                    writer.print(" & ");
                    writer.print(String.format("%.2f", avg[0] * 100));
                    writer.print(" & ");
                    writer.print(String.format("%.2f", avg[1] * 100));
                    writer.print(" & ");
                    writer.print(String.format("%.2f", avg[2] * 100));
                } else {
                    writer.print(" & --- & --- & ---");
                }
            }

            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        writer.println("}");
        writer.println("\\end{table}");
    }

    /**
     * Option 2: Best method per dataset and classifier
     */
    private static void generateOption2_BestPerDataset(PrintWriter writer,
            Map<String, Map<String, Map<String, MetricValues>>> allDatasets) {

        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Best Performing Method for Each Dataset and Classifier}");
        writer.println("\\label{tab:summary_best}");
        writer.println("\\begin{tabular}{l|ccc}");
        writer.println("\\toprule");
        writer.println("\\textbf{Dataset} & \\textbf{J48} & \\textbf{RandomForest} & \\textbf{NaiveBayes} \\\\");
        writer.println("\\midrule");

        for (String[] dataset : DATASETS) {
            String datasetName = dataset[0];
            if (!allDatasets.containsKey(datasetName)) continue;

            Map<String, Map<String, MetricValues>> results = allDatasets.get(datasetName);

            writer.print(datasetName);

            for (String classifier : CLASSIFIERS) {
                String bestMethod = "";
                double bestValue = -1;

                for (String method : METHODS) {
                    if (results.containsKey(method)) {
                        MetricValues mv = results.get(method).get(classifier);
                        double fmeasure = calculateMacroAverage(mv.fmeasureValues);
                        if (fmeasure > bestValue) {
                            bestValue = fmeasure;
                            bestMethod = method;
                        }
                    }
                }

                writer.print(" & ");
                writer.print(formatMethodName(bestMethod) + " (" + String.format("%.2f", bestValue * 100) + ")");
            }

            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        writer.println("\\end{table}");
    }

    /**
     * Option 3: Ranking frequency across all experiments (excluding Original)
     */
    private static void generateOption3_RankingFrequency(PrintWriter writer,
            Map<String, Map<String, Map<String, MetricValues>>> allDatasets) {

        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Ranking Frequency of Privacy-Preserving Methods Across All Datasets and Classifiers}");
        writer.println("\\label{tab:summary_ranking}");
        // Add configurable width comment
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            double widthPercent = Double.parseDouble(TABLE_WIDTH_PERCENTAGE) * 100;
            writer.println("% Table width: " + String.format("%.0f", widthPercent) + "% - You can modify this value (e.g., 0.90, 0.85, 0.80)");
            writer.println("\\resizebox{" + TABLE_WIDTH_PERCENTAGE + "\\textwidth}{!}{%");
        }
        writer.println("\\begin{tabular}{c|l|ccccc|c}");
        writer.println("\\toprule");
        writer.println("\\textbf{Rank} & \\textbf{Method} & \\textbf{1st} & \\textbf{2nd} & \\textbf{3rd} & \\textbf{4th} & \\textbf{5th} & \\textbf{Avg Rank} \\\\");
        writer.println("\\midrule");

        Map<String, int[]> rankCounts = new HashMap<>();
        for (String method : METHODS) {
            if (!method.equals("Original")) {
                rankCounts.put(method, new int[5]); // ranks 1-5 (excluding Original)
            }
        }

        // Count rankings (excluding Original)
        for (Map<String, Map<String, MetricValues>> datasetResults : allDatasets.values()) {
            for (String classifier : CLASSIFIERS) {
                List<ValueMethodPair> pairs = new ArrayList<>();

                for (String method : METHODS) {
                    // Skip Original method
                    if (method.equals("Original")) continue;

                    if (datasetResults.containsKey(method)) {
                        MetricValues mv = datasetResults.get(method).get(classifier);
                        double fmeasure = calculateMacroAverage(mv.fmeasureValues);
                        pairs.add(new ValueMethodPair(fmeasure, method));
                    }
                }

                pairs.sort((a, b) -> Double.compare(b.value, a.value));

                for (int i = 0; i < pairs.size(); i++) {
                    String method = pairs.get(i).method;
                    rankCounts.get(method)[i]++;
                }
            }
        }

        // Calculate average ranks and create sorted list
        List<MethodRankPair> methodRanks = new ArrayList<>();
        for (String method : METHODS) {
            if (method.equals("Original")) continue;

            int[] counts = rankCounts.get(method);
            int totalRank = 0;
            int totalCount = 0;
            for (int i = 0; i < 5; i++) {
                totalRank += (i + 1) * counts[i];
                totalCount += counts[i];
            }

            double avgRank = totalCount > 0 ? (double) totalRank / totalCount : 0;
            methodRanks.add(new MethodRankPair(method, avgRank, counts));
        }

        // Sort by average rank (lower is better)
        methodRanks.sort((a, b) -> Double.compare(a.avgRank, b.avgRank));

        // Write data rows sorted by overall rank
        int overallRank = 1;
        for (MethodRankPair mrp : methodRanks) {
            writer.print(overallRank);
            writer.print(" & ");
            writer.print(formatMethodName(mrp.method));

            for (int i = 0; i < 5; i++) {
                writer.print(" & " + mrp.counts[i]);
            }

            writer.print(" & " + String.format("%.2f", mrp.avgRank));
            writer.println(" \\\\");
            overallRank++;
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            writer.println("}");
        }
        writer.println("\\end{table}");
    }

    /**
     * Helper class to store method with its average rank and counts
     */
    private static class MethodRankPair {
        String method;
        double avgRank;
        int[] counts;

        MethodRankPair(String method, double avgRank, int[] counts) {
            this.method = method;
            this.avgRank = avgRank;
            this.counts = counts;
        }
    }

    /**
     * Option 4: Side-by-side dataset comparison (F-Measure only)
     */
    private static void generateOption4_SideBySide(PrintWriter writer,
            Map<String, Map<String, Map<String, MetricValues>>> allDatasets) {

        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{F-Measure Comparison Across All Datasets (RandomForest)}");
        writer.println("\\label{tab:summary_sidebyside}");
        writer.println("\\begin{tabular}{l|cccc}");
        writer.println("\\toprule");
        writer.print("\\textbf{Method}");
        for (String[] dataset : DATASETS) {
            writer.print(" & \\textbf{" + dataset[0] + "}");
        }
        writer.println(" \\\\");
        writer.println("\\midrule");

        for (String method : METHODS) {
            writer.print(formatMethodName(method));

            for (String[] dataset : DATASETS) {
                String datasetName = dataset[0];
                if (allDatasets.containsKey(datasetName)) {
                    Map<String, Map<String, MetricValues>> results = allDatasets.get(datasetName);
                    if (results.containsKey(method)) {
                        MetricValues mv = results.get(method).get("RandomForest");
                        double fmeasure = calculateMacroAverage(mv.fmeasureValues);
                        writer.print(" & " + String.format("%.2f", fmeasure * 100));
                    } else {
                        writer.print(" & ---");
                    }
                } else {
                    writer.print(" & ---");
                }
            }

            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        writer.println("\\end{table}");
    }

    /**
     * Parse CSV file and extract metrics
     */
    private static Map<String, Map<String, MetricValues>> parseCSV(String filePath) throws Exception {
        Map<String, Map<String, MetricValues>> results = new HashMap<>();

        // Initialize structure
        for (String method : METHODS) {
            results.put(method, new HashMap<>());
            for (String classifier : CLASSIFIERS) {
                results.get(method).put(classifier, new MetricValues());
            }
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Skip header

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 5) continue;

                String method = parts[0].trim();
                String classifier = parts[1].trim();
                String metric = parts[2].trim();
                String classLabel = parts[3].trim();
                double value = Double.parseDouble(parts[4].trim());

                if (!results.containsKey(method)) continue;

                MetricValues mv = results.get(method).get(classifier);

                // Collect per-class metrics for averaging
                if (metric.equals("Precision") && !classLabel.isEmpty()) {
                    mv.precisionValues.add(value);
                } else if (metric.equals("Recall") && !classLabel.isEmpty()) {
                    mv.recallValues.add(value);
                } else if (metric.equals("F-Measure") && !classLabel.isEmpty()) {
                    mv.fmeasureValues.add(value);
                }
            }
        }

        return results;
    }

    /**
     * Write LaTeX table (without document preamble) - for inclusion in main document
     * Includes superscript ranking for each metric
     */
    private static void writeLatexTable(PrintWriter writer,
                                        Map<String, Map<String, MetricValues>> results,
                                        String datasetDisplayName,
                                        String datasetPrefix) {
        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Classification Results for " + datasetDisplayName + "}");
        writer.println("\\label{tab:" + datasetPrefix + "_results}");
        // Add configurable width comment
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            double widthPercent = Double.parseDouble(TABLE_WIDTH_PERCENTAGE) * 100;
            writer.println("% Table width: " + String.format("%.0f", widthPercent) + "% - You can modify this value (e.g., 0.90, 0.85, 0.80)");
            writer.println("\\resizebox{" + TABLE_WIDTH_PERCENTAGE + "\\textwidth}{!}{%");
        }
        writer.println("\\begin{tabular}{l|ccc|ccc|ccc}");
        writer.println("\\toprule");

        // Header row 1: Classifier names
        writer.println("\\multirow{2}{*}{\\textbf{Method}} & " +
                      "\\multicolumn{3}{c|}{\\textbf{J48}} & " +
                      "\\multicolumn{3}{c|}{\\textbf{RandomForest}} & " +
                      "\\multicolumn{3}{c}{\\textbf{NaiveBayes}} \\\\");

        // Header row 2: Metric names (abbreviated)
        writer.println("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}");
        writer.println(" & M-Prec & M-F1 & M-Rec & " +
                      "M-Prec & M-F1 & M-Rec & " +
                      "M-Prec & M-F1 & M-Rec \\\\");
        writer.println("\\midrule");

        // First pass: collect all values and calculate rankings
        Map<String, List<ValueMethodPair>> valuesByMetric = new HashMap<>();

        for (String method : METHODS) {
            if (!results.containsKey(method)) continue;
            Map<String, MetricValues> methodResults = results.get(method);

            for (String classifier : CLASSIFIERS) {
                MetricValues mv = methodResults.get(classifier);
                double precision = calculateMacroAverage(mv.precisionValues);
                double fmeasure = calculateMacroAverage(mv.fmeasureValues);
                double recall = calculateMacroAverage(mv.recallValues);

                String precKey = classifier + "_precision";
                String fmKey = classifier + "_fmeasure";
                String recKey = classifier + "_recall";

                valuesByMetric.putIfAbsent(precKey, new ArrayList<>());
                valuesByMetric.putIfAbsent(fmKey, new ArrayList<>());
                valuesByMetric.putIfAbsent(recKey, new ArrayList<>());

                valuesByMetric.get(precKey).add(new ValueMethodPair(precision, method));
                valuesByMetric.get(fmKey).add(new ValueMethodPair(fmeasure, method));
                valuesByMetric.get(recKey).add(new ValueMethodPair(recall, method));
            }
        }

        // Calculate ranks for each metric (higher is better)
        Map<String, Map<String, Integer>> rankings = new HashMap<>();
        for (Map.Entry<String, List<ValueMethodPair>> entry : valuesByMetric.entrySet()) {
            String metricKey = entry.getKey();
            List<ValueMethodPair> pairs = entry.getValue();

            // Sort descending (higher values get better rank)
            pairs.sort((a, b) -> Double.compare(b.value, a.value));

            Map<String, Integer> methodRanks = new HashMap<>();
            for (int i = 0; i < pairs.size(); i++) {
                methodRanks.put(pairs.get(i).method, i + 1);
            }
            rankings.put(metricKey, methodRanks);
        }

        // Second pass: write data rows with rankings
        for (String method : METHODS) {
            if (!results.containsKey(method)) continue;

            Map<String, MetricValues> methodResults = results.get(method);

            writer.print(formatMethodName(method));

            for (String classifier : CLASSIFIERS) {
                MetricValues mv = methodResults.get(classifier);

                // Calculate macro-averages
                double precision = calculateMacroAverage(mv.precisionValues);
                double fmeasure = calculateMacroAverage(mv.fmeasureValues);
                double recall = calculateMacroAverage(mv.recallValues);

                String precKey = classifier + "_precision";
                String fmKey = classifier + "_fmeasure";
                String recKey = classifier + "_recall";

                int precRank = rankings.get(precKey).get(method);
                int fmRank = rankings.get(fmKey).get(method);
                int recRank = rankings.get(recKey).get(method);

                writer.print(" & ");
                writer.print(formatValueWithRank(precision * 100, precRank));
                writer.print(" & ");
                writer.print(formatValueWithRank(fmeasure * 100, fmRank));
                writer.print(" & ");
                writer.print(formatValueWithRank(recall * 100, recRank));
            }

            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            writer.println("}");
        }
        writer.println("\\end{table}");
    }

    /**
     * Format value with superscript rank
     */
    private static String formatValueWithRank(double value, int rank) {
        String formatted = String.format("%.2f", value);
        return formatted + "\\textsuperscript{" + rank + "}";
    }

    /**
     * Helper class to pair value with method name for ranking
     */
    private static class ValueMethodPair {
        double value;
        String method;

        ValueMethodPair(double value, String method) {
            this.value = value;
            this.method = method;
        }
    }

    /**
     * Generate LaTeX file with table (with document preamble - deprecated)
     */
    private static void generateLatexFile(Map<String, Map<String, MetricValues>> results,
                                          String datasetName, String outputFile) throws Exception {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            // Write LaTeX document
            writer.println("\\documentclass{article}");
            writer.println("\\usepackage{booktabs}");
            writer.println("\\usepackage{multirow}");
            writer.println("\\usepackage{geometry}");
            writer.println("\\geometry{landscape, margin=1in}");
            writer.println();
            writer.println("\\begin{document}");
            writer.println();

            // Write table
            writeLatexTable(writer, results, formatDatasetName(datasetName), datasetName);

            writer.println();
            writer.println("\\end{document}");
        }
    }

    /**
     * Format method name for LaTeX
     */
    private static String formatMethodName(String method) {
        switch (method) {
            case "Original":
                return "Original";
            case "PrivacyGuard":
                return "PrivacyGuard (VWC)";
            case "Equal_Width_Binning":
                return "Equal Width Binning";
            case "kmeans":
                return "k-means (k=1000)";
            case "k_anonymity":
                return "k-anonymity (k=5)";
            case "LaplaceDP":
                return "Laplace DP ($\\epsilon$=1.0)";
            default:
                return method;
        }
    }

    /**
     * Format dataset name for LaTeX
     */
    private static String formatDatasetName(String dataset) {
        return dataset.replace("_", " ").replace("train", "Dataset");
    }

    /**
     * Calculate macro-average from list of values
     */
    private static double calculateMacroAverage(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    /**
     * Pad string to the right with spaces
     */
    private static String padRight(String str, int length) {
        if (str.length() >= length) {
            return str.substring(0, length);
        }
        return str + " ".repeat(length - str.length());
    }

    /**
     * Helper class to store metric values
     */
    private static class MetricValues {
        List<Double> precisionValues = new ArrayList<>();
        List<Double> recallValues = new ArrayList<>();
        List<Double> fmeasureValues = new ArrayList<>();
    }

    /**
     * Generate all privacy results LaTeX tables in one file
     */
    public static void generatePrivacyResultsTables() throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║       Generating Privacy Results LaTeX Tables                ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        String outputFile = LATEX_OUTPUT_DIR + "privacy_results_tables.tex";

        // Create output directory
        new File(LATEX_OUTPUT_DIR).mkdirs();

        // Get all privacy result CSV files
        File dir = new File(PRIVACY_DIR);
        File[] files = dir.listFiles((d, name) -> name.endsWith("_privacy_results.csv"));

        if (files == null || files.length == 0) {
            System.out.println("✗ No privacy results files found in " + PRIVACY_DIR);
            System.out.println("  Please run privacy attack evaluation first.");
            return;
        }

        // Parse all CSV files
        Map<String, Map<String, PrivacyMetrics>> privacyResults = new HashMap<>();

        for (File file : files) {
            System.out.println("Processing: " + file.getName());
            parsePrivacyCSV(file.getAbsolutePath(), privacyResults);
        }

        // Write all tables to one file
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            // Generate Re-identification tables for each dataset
            for (String[] dataset : DATASETS) {
                String displayName = dataset[0];
                String datasetKey = dataset[0];

                if (privacyResults.containsKey(datasetKey)) {
                    writePrivacyReIdentificationTable(writer, privacyResults.get(datasetKey), displayName);
                    writer.println();
                    writer.println();
                }
            }

            // Generate Linkage tables for each dataset
            for (String[] dataset : DATASETS) {
                String displayName = dataset[0];
                String datasetKey = dataset[0];

                if (privacyResults.containsKey(datasetKey)) {
                    writePrivacyLinkageTable(writer, privacyResults.get(datasetKey), displayName);
                    writer.println();
                    writer.println();
                }
            }
        }

        System.out.println();
        System.out.println("✓ Privacy results LaTeX tables generated successfully!");
        System.out.println("  Output: " + outputFile);
        System.out.println("  Note: No document preamble included - ready for inclusion");
        System.out.println();
    }

    /**
     * Parse privacy results CSV file
     */
    private static void parsePrivacyCSV(String filePath,
                                       Map<String, Map<String, PrivacyMetrics>> results) throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // Skip header

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 5) continue;

                String dataset = parts[0].trim();
                String method = parts[1].trim();
                String attackType = parts[2].trim();
                String metric = parts[3].trim();
                double value = Double.parseDouble(parts[4].trim());

                // Ensure dataset exists in results
                results.putIfAbsent(dataset, new HashMap<>());

                // Ensure method exists for this dataset
                results.get(dataset).putIfAbsent(method, new PrivacyMetrics());

                PrivacyMetrics pm = results.get(dataset).get(method);

                // Store metrics based on attack type
                if (attackType.equals("Re-identification")) {
                    switch (metric) {
                        case "Re-identification Rate":
                            pm.reIdentificationRate = value;
                            break;
                        case "Top-1 Accuracy":
                            pm.top1Accuracy = value;
                            break;
                        case "Top-5 Accuracy":
                            pm.top5Accuracy = value;
                            break;
                        case "Privacy Risk Score":
                            pm.privacyRiskScore = value;
                            break;
                    }
                } else if (attackType.equals("Linkage")) {
                    switch (metric) {
                        case "Knowledge_25%_Correct_Match_Rate":
                            pm.linkage25CorrectRate = value;
                            break;
                        case "Knowledge_50%_Correct_Match_Rate":
                            pm.linkage50CorrectRate = value;
                            break;
                        case "Knowledge_75%_Correct_Match_Rate":
                            pm.linkage75CorrectRate = value;
                            break;
                        case "Knowledge_25%_Avg_Group_Size":
                            pm.linkage25GroupSize = value;
                            break;
                        case "Knowledge_50%_Avg_Group_Size":
                            pm.linkage50GroupSize = value;
                            break;
                        case "Knowledge_75%_Avg_Group_Size":
                            pm.linkage75GroupSize = value;
                            break;
                    }
                }
            }
        }
    }

    /**
     * Write Re-identification attack results table
     */
    private static void writePrivacyReIdentificationTable(PrintWriter writer,
                                                          Map<String, PrivacyMetrics> methodResults,
                                                          String datasetName) {
        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Re-identification Attack Results for " + datasetName + "}");
        writer.println("\\label{tab:" + datasetName.toLowerCase().replace("-", "") + "_reidentification}");
        // Add configurable width comment
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            double widthPercent = Double.parseDouble(TABLE_WIDTH_PERCENTAGE) * 100;
            writer.println("% Table width: " + String.format("%.0f", widthPercent) + "% - You can modify this value (e.g., 0.90, 0.85, 0.80)");
            writer.println("\\resizebox{" + TABLE_WIDTH_PERCENTAGE + "\\textwidth}{!}{%");
        }
        writer.println("\\begin{tabular}{l|cccc}");
        writer.println("\\toprule");
        writer.println("\\textbf{Method} & \\textbf{Re-ID Rate} & \\textbf{Top-1 Acc.} & \\textbf{Top-5 Acc.} & \\textbf{Risk Score} \\\\");
        writer.println("\\midrule");

        // Write data for each method
        for (String method : METHODS) {
            if (method.equals("Original")) continue; // Skip Original for privacy results

            PrivacyMetrics pm = methodResults.get(method);
            if (pm == null) continue;

            writer.print(formatMethodName(method));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.reIdentificationRate));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.top1Accuracy));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.top5Accuracy));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.privacyRiskScore));
            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            writer.println("}");
        }
        writer.println("\\end{table}");
    }

    /**
     * Write Linkage attack results table
     */
    private static void writePrivacyLinkageTable(PrintWriter writer,
                                                Map<String, PrivacyMetrics> methodResults,
                                                String datasetName) {
        writer.println("\\begin{table}[htbp]");
        writer.println("\\centering");
        writer.println("\\caption{Linkage Attack Results for " + datasetName + "}");
        writer.println("\\label{tab:" + datasetName.toLowerCase().replace("-", "") + "_linkage}");
        // Add configurable width comment
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            double widthPercent = Double.parseDouble(TABLE_WIDTH_PERCENTAGE) * 100;
            writer.println("% Table width: " + String.format("%.0f", widthPercent) + "% - You can modify this value (e.g., 0.90, 0.85, 0.80)");
            writer.println("\\resizebox{" + TABLE_WIDTH_PERCENTAGE + "\\textwidth}{!}{%");
        }
        writer.println("\\begin{tabular}{l|ccc|ccc}");
        writer.println("\\toprule");
        writer.println("\\multirow{2}{*}{\\textbf{Method}} & " +
                      "\\multicolumn{3}{c|}{\\textbf{Correct Match Rate}} & " +
                      "\\multicolumn{3}{c}{\\textbf{Avg. Group Size}} \\\\");
        writer.println("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}");
        writer.println(" & 25\\% & 50\\% & 75\\% & 25\\% & 50\\% & 75\\% \\\\");
        writer.println("\\midrule");

        // Write data for each method
        for (String method : METHODS) {
            if (method.equals("Original")) continue; // Skip Original for privacy results

            PrivacyMetrics pm = methodResults.get(method);
            if (pm == null) continue;

            writer.print(formatMethodName(method));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.linkage25CorrectRate));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.linkage50CorrectRate));
            writer.print(" & ");
            writer.print(String.format("%.4f", pm.linkage75CorrectRate));
            writer.print(" & ");
            writer.print(String.format("%.2f", pm.linkage25GroupSize));
            writer.print(" & ");
            writer.print(String.format("%.2f", pm.linkage50GroupSize));
            writer.print(" & ");
            writer.print(String.format("%.2f", pm.linkage75GroupSize));
            writer.println(" \\\\");
        }

        writer.println("\\bottomrule");
        writer.println("\\end{tabular}");
        if (!TABLE_WIDTH_PERCENTAGE.isEmpty()) {
            writer.println("}");
        }
        writer.println("\\end{table}");
    }

    /**
     * Helper class to store privacy metrics
     */
    private static class PrivacyMetrics {
        // Re-identification metrics
        double reIdentificationRate = 0.0;
        double top1Accuracy = 0.0;
        double top5Accuracy = 0.0;
        double privacyRiskScore = 0.0;

        // Linkage metrics
        double linkage25CorrectRate = 0.0;
        double linkage50CorrectRate = 0.0;
        double linkage75CorrectRate = 0.0;
        double linkage25GroupSize = 0.0;
        double linkage50GroupSize = 0.0;
        double linkage75GroupSize = 0.0;
    }

    /**
     * Main method for standalone execution
     */
    public static void main(String[] args) throws Exception {
        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║           LaTeX Table Generator - Standalone                 ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");

        if (args.length > 0) {
            // Generate table for specific dataset
            generateLatexTable(args[0]);
        } else {
            // Generate all tables in one file
            generateAllTablesInOneFile();
        }
    }
}
