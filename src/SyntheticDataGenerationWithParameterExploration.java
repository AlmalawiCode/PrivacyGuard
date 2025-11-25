/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package privacyguard;

import privacyguard.ConfigLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Scanner;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import wekaknnvwc.ProfTools;
import wekaknnvwc.VWC;

/**
 * Enhanced Synthetic Data Generation with Parameter Space Exploration
 * 
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 * Enhanced by Manus AI with comprehensive parameter exploration framework
 */
public class SyntheticDataGenerationWithParameterExploration {
    
    // Beta value loaded from configuration

    // Parameter exploration configuration - can be overridden by menu
    private static int[] CLUSTER_SIZE_OPTIONS = {5, 10, 15, 20, 25, 30, 35};

    // Results storage for all configurations
    private static Map<String, List<ConfigurationResult>> allConfigurationResults = new HashMap<>();
    
    /**
     * Class to store results for a specific parameter configuration
     */
    public static class ConfigurationResult {
        public int maxClusterSize;
        public double betaPercentage;
        public EvaluationResults evaluation;
        public long processingTimeMs;
        
        // Added fields for cluster statistics
        public Map<Integer, Integer> clusterCountsPerFeature = new HashMap<>();
        public double avgClustersPerFeature = 0.0;
        public double clusterCountVariance = 0.0;
        public double clusterCountStdDev = 0.0;
        public int minClusters = Integer.MAX_VALUE;
        public int maxClusters = 0;
        public int totalClusters = 0;
        
        // Added fields for all three variance metrics
        public double rawVariance = 0.0;
        public double normalizedVariance = 0.0;
        public double coefficientOfVariation = 0.0;
        
        public ConfigurationResult(int maxClusterSize, double betaPercentage) {
            this.maxClusterSize = maxClusterSize;
            this.betaPercentage = betaPercentage;
            this.evaluation = new EvaluationResults();
        }
    }
    
    /**
     * Class to store evaluation metrics for each configuration
     */
    public static class EvaluationResults {
        public Map<Integer, Double> utilityPreservation = new HashMap<>();
        public Map<Integer, Double> anonymityStrength = new HashMap<>();
        public Map<Integer, Double> combinedMetric = new HashMap<>();
        public Map<Integer, Double> alphaValues = new HashMap<>();
        public double overallUtility;
        public double overallAnonymity;
        public double overallCombined;
        public int totalFeatures;
        public int totalInstances;
        
        // Clustering statistics
        public Map<Integer, Integer> clusterCounts = new HashMap<>();
        public Map<Integer, Double> originalVariances = new HashMap<>();
        public Map<Integer, Double> withinClusterVariances = new HashMap<>();
    }

    /**
     * Menu wrapper for calling from MenuDrivenInterface
     * @param scanner Shared scanner from main menu
     */
    public static void showParameterExplorationMenu(Scanner scanner) {
        try {
            runParameterExploration(scanner);
        } catch (Exception e) {
            System.err.println("\n✗ Error in parameter exploration: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        runParameterExploration(scanner);
        scanner.close();
    }

    private static void runParameterExploration(Scanner scanner) throws Exception {
        ProfTools profTools = new ProfTools();

        // Directory to save parameter exploration results (LaTeX + results files only)
        String outputDirectory = "output/6_parameter_exploration/";
        new File(outputDirectory).mkdirs();

        // File paths for the datasets - using ConfigLoader
        String[] filePaths = ConfigLoader.getDatasetPaths();
        String[] datasetNames = {"bot_loT", "CICIoT2023", "MQTTset", "Edge-IIoTset"};

        // Display menu
        System.out.println("=".repeat(60));
        System.out.println("  VWC Parameter Exploration - Dataset Selection Menu");
        System.out.println("=".repeat(60));
        System.out.println("\nAvailable Datasets:");
        for (int i = 0; i < filePaths.length; i++) {
            System.out.printf("  %d. %s%n", i + 1, datasetNames[i]);
        }
        System.out.println("  5. All datasets");
        System.out.println("  0. Back to Main Menu");
        System.out.println();

        // Get dataset selection
        System.out.print("Select dataset(s) to process (0-5): ");
        int datasetChoice = scanner.nextInt();

        if (datasetChoice == 0) {
            System.out.println("\n← Returning to Main Menu...\n");
            return;
        }

        String[] selectedPaths;
        if (datasetChoice == 5) {
            selectedPaths = filePaths;
            System.out.println("Selected: All datasets");
        } else if (datasetChoice >= 1 && datasetChoice <= 4) {
            selectedPaths = new String[]{filePaths[datasetChoice - 1]};
            System.out.println("Selected: " + datasetNames[datasetChoice - 1]);
        } else {
            System.err.println("Invalid choice. Returning to main menu.");
            return;
        }

        // Get s_Max values to test
        System.out.println("\n" + "-".repeat(60));
        System.out.println("s_Max (Max Cluster Size) Configuration");
        System.out.println("-".repeat(60));
        System.out.println("Default values: 5, 10, 15, 20, 25, 30, 35");
        System.out.println("\nOptions:");
        System.out.println("  1. Use default values (5, 10, 15, 20, 25, 30, 35)");
        System.out.println("  2. Enter custom values");
        System.out.print("\nSelect option (1-2): ");
        int sMaxOption = scanner.nextInt();

        if (sMaxOption == 2) {
            System.out.print("Enter s_Max values separated by commas (e.g., 5,10,15,20,25,30,35): ");
            String valuesInput = scanner.next();
            String[] valueParts = valuesInput.split(",");
            List<Integer> clusterSizes = new ArrayList<>();
            for (String part : valueParts) {
                clusterSizes.add(Integer.parseInt(part.trim()));
            }
            CLUSTER_SIZE_OPTIONS = clusterSizes.stream().mapToInt(Integer::intValue).toArray();
            System.out.println("\ns_Max values to test: " + clusterSizes);
        } else {
            System.out.println("\nUsing default s_Max values: [5, 10, 15, 20, 25, 30, 35]");
        }
        System.out.println();

        // Confirm before running
        System.out.println("\nPress Enter to start processing...");
        System.out.flush();
        scanner.nextLine(); // consume newline from previous nextInt()
        scanner.nextLine(); // wait for user to press enter

        // Process each selected dataset with parameter exploration
        for (String filePath : selectedPaths) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("PROCESSING DATASET: " + new File(filePath).getName());
            System.out.println("=".repeat(80));

            // Load the dataset (supports both CSV and ARFF formats)
            Instances originalData = loadDataset(filePath);
            String datasetName = new File(filePath).getName();

            // Initialize results storage for this dataset
            allConfigurationResults.put(datasetName, new ArrayList<>());

            // Explore different cluster size configurations
            exploreParameterSpace(originalData, datasetName, profTools, outputDirectory);

            // Find optimal configuration
            ConfigurationResult optimalConfig = findOptimalConfiguration(datasetName);
            System.out.println("\nOptimal Configuration: maxClusterSize=" + optimalConfig.maxClusterSize +
                             ", beta=" + String.format("%.1f%%", optimalConfig.betaPercentage));

            // Generate results report for this dataset
            generateParameterExplorationReport(datasetName, outputDirectory);
        }

        // Generate final comparison report across all datasets
        generateFinalComparisonReport(outputDirectory);

        // Generate LaTeX table for publication
        generateLatexTable(outputDirectory);

        System.out.println("\n" + "=".repeat(80));
        System.out.println("PARAMETER EXPLORATION COMPLETE");
        System.out.println("=".repeat(80));
        System.out.println("\nResults saved to: " + outputDirectory);
        System.out.println("  - cluster_size_analysis_table.tex (LaTeX table)");
        System.out.println("  - final_parameter_comparison_report.txt");
        System.out.println("  - [dataset]_parameter_exploration_report.txt (per dataset)");
        System.out.println();
    }

    /**
     * Explore parameter space by testing different cluster size configurations
     */
    public static void exploreParameterSpace(Instances originalData, String datasetName, 
                                           ProfTools profTools, String outputDirectory) throws Exception {
        
        System.out.println("Starting parameter space exploration...");
        System.out.printf("Testing %d different cluster size configurations (beta = 30%% of s_Max)\n",
                         CLUSTER_SIZE_OPTIONS.length);
        System.out.println("-".repeat(80));
        
        List<ConfigurationResult> results = allConfigurationResults.get(datasetName);
        
        for (int i = 0; i < CLUSTER_SIZE_OPTIONS.length; i++) {
            int maxClusterSize = CLUSTER_SIZE_OPTIONS[i];

            int beta = (int) Math.round(maxClusterSize * 0.30);
            if (beta < 2) beta = 2;
            System.out.printf("[%s] Configuration %d/%d: Testing maxClusterSize=%d, beta=%d\n",
                            datasetName, i + 1, CLUSTER_SIZE_OPTIONS.length, maxClusterSize, beta);
            
            long startTime = System.currentTimeMillis();
            
            // Create configuration result
            ConfigurationResult configResult = new ConfigurationResult(maxClusterSize, beta);
            configResult.evaluation.totalInstances = originalData.numInstances();
            
            // Test this configuration
            Instances testData = new Instances(originalData);
            generateSyntheticDataWithEvaluation(testData, profTools, configResult, maxClusterSize);
            
            // Calculate overall metrics
            calculateOverallMetrics(configResult.evaluation);
            
            configResult.processingTimeMs = System.currentTimeMillis() - startTime;
            
            // Store results
            results.add(configResult);
            
            // Print configuration summary with added cluster statistics
            System.out.printf("  Results: Utility=%.3f, Anonymity=%.1f, Combined=%.3f, AvgClusters=%.2f (%.2fs)\n",
                            configResult.evaluation.overallUtility,
                            configResult.evaluation.overallAnonymity,
                            configResult.evaluation.overallCombined,
                            configResult.avgClustersPerFeature,
                            configResult.processingTimeMs / 1000.0);
            
            // Print all variance metrics
            System.out.printf("  Variance Metrics: Raw=%.4f, Normalized=%.4f, CV=%.4f\n",
                            configResult.rawVariance,
                            configResult.normalizedVariance,
                            configResult.coefficientOfVariation);
        }
        
        System.out.println("-".repeat(80));
        System.out.println("Parameter exploration completed for " + datasetName);
    }

    /**
     * Find optimal configuration based on combined metric
     */
    public static ConfigurationResult findOptimalConfiguration(String datasetName) {
        List<ConfigurationResult> results = allConfigurationResults.get(datasetName);
        
        ConfigurationResult optimal = results.stream()
            .max((r1, r2) -> Double.compare(r1.evaluation.overallCombined, r2.evaluation.overallCombined))
            .orElse(results.get(0));
        
        System.out.println("\nOPTIMAL CONFIGURATION FOUND:");
        System.out.printf("  Max Cluster Size: %d\n", optimal.maxClusterSize);
        System.out.printf("  Beta: %.1f%%\n", optimal.betaPercentage);
        System.out.printf("  Overall Utility: %.3f\n", optimal.evaluation.overallUtility);
        System.out.printf("  Overall Anonymity: %.1f\n", optimal.evaluation.overallAnonymity);
        System.out.printf("  Overall Combined: %.3f\n", optimal.evaluation.overallCombined);
        System.out.printf("  Average Clusters per Feature: %.2f\n", optimal.avgClustersPerFeature);
        System.out.printf("  Variance Metrics: Raw=%.4f, Normalized=%.4f, CV=%.4f\n",
                        optimal.rawVariance,
                        optimal.normalizedVariance,
                        optimal.coefficientOfVariation);
        
        return optimal;
    }

    /**
     * Generate synthetic data with comprehensive evaluation for parameter exploration
     */
    public static Instances generateSyntheticDataWithEvaluation(Instances data, ProfTools profTools, 
                                                               ConfigurationResult configResult, int maxClusterSize) throws Exception {
        // Determine if there is a class attribute
        int classIndex = data.classIndex();
        boolean hasClassAttribute = classIndex != -1;

        // Save the class attribute if it exists
        Instances originalDataWithClass = null;
        if (hasClassAttribute) {
            originalDataWithClass = new Instances(data);
        }

        int featureCount = 0;
        
        // Iterate over each attribute to perform clustering and evaluation
        for (int featureIndex = 0; featureIndex < data.numAttributes(); featureIndex++) {
            // Skip the class attribute if it exists
            if (featureIndex == classIndex) continue;
            
            featureCount++;

            // Calculate original feature statistics
            double[] originalValues = extractFeatureValues(data, featureIndex);
            double originalMean = calculateMean(originalValues);
            double originalVariance = calculateVariance(originalValues, originalMean);
            configResult.evaluation.originalVariances.put(featureIndex, originalVariance);

            // Create single feature dataset for clustering
            Instances singleFeatureData = createSingleFeatureDataset(data, featureIndex);

            // Perform VWC clustering with specified parameters
            ClusteringResults clusterResults = performVWCClusteringWithParams(singleFeatureData, maxClusterSize);
            
            // Collect cluster statistics
            collectClusterStatistics(clusterResults, configResult, featureIndex);
            
            // Store clustering statistics
            configResult.evaluation.clusterCounts.put(featureIndex, clusterResults.numClusters);
            configResult.evaluation.withinClusterVariances.put(featureIndex, clusterResults.totalWithinClusterVariance);

            // Calculate evaluation metrics for this feature
            double utilityPreservation = calculateUtilityPreservation(
                originalVariance, clusterResults.totalWithinClusterVariance);
            double anonymityStrength = calculateAnonymityStrength(
                clusterResults, data.numInstances());

            // Set alpha value based on feature characteristics
            double alpha = determineAlphaValue(data.attribute(featureIndex).name());
            configResult.evaluation.alphaValues.put(featureIndex, alpha);

            double combinedMetric = calculateCombinedMetric(
                utilityPreservation, anonymityStrength, alpha, data.numInstances());

            // Store evaluation results
            configResult.evaluation.utilityPreservation.put(featureIndex, utilityPreservation);
            configResult.evaluation.anonymityStrength.put(featureIndex, anonymityStrength);
            configResult.evaluation.combinedMetric.put(featureIndex, combinedMetric);

            // Replace original feature values with cluster IDs
            int[] clusterAssignments = clusterResults.assignments;
            for (int i = 0; i < data.numInstances(); i++) {
                data.instance(i).setValue(featureIndex, clusterAssignments[i]);
            }
        }
        
        configResult.evaluation.totalFeatures = featureCount;
        
        // Finalize cluster statistics with all three variance metrics
        finalizeClusterStatistics(configResult);

        // Reattach the original class attribute if it exists
        if (hasClassAttribute) {
            for (int i = 0; i < data.numInstances(); i++) {
                data.instance(i).setClassValue(originalDataWithClass.instance(i).classValue());
            }
        }

        return data;
    }
    
    /**
     * Collect cluster statistics during evaluation
     */
    public static void collectClusterStatistics(ClusteringResults clusterResults, 
                                              ConfigurationResult configResult,
                                              int featureIndex) {
        // Store cluster count for this feature
        int clusterCount = clusterResults.numClusters;
        configResult.clusterCountsPerFeature.put(featureIndex, clusterCount);
        
        // Update min/max cluster counts
        configResult.minClusters = Math.min(configResult.minClusters, clusterCount);
        configResult.maxClusters = Math.max(configResult.maxClusters, clusterCount);
        configResult.totalClusters += clusterCount;
    }
    
    /**
     * Calculate final cluster statistics after processing all features
     * Includes all three variance metrics
     */
    public static void finalizeClusterStatistics(ConfigurationResult configResult) {
        if (configResult.evaluation.totalFeatures > 0) {
            // Calculate average clusters per feature
            configResult.avgClustersPerFeature = 
                (double) configResult.totalClusters / configResult.evaluation.totalFeatures;
            
            // Calculate all three variance metrics
            double sumSquaredDiff = 0.0;
            double sumSquaredNormalizedDiff = 0.0;
            
            for (Integer clusterCount : configResult.clusterCountsPerFeature.values()) {
                // For raw variance
                double diff = clusterCount - configResult.avgClustersPerFeature;
                sumSquaredDiff += Math.pow(diff, 2);
                
                // For normalized variance
                double normalizedDiff = diff / Math.max(1.0, configResult.avgClustersPerFeature);
                sumSquaredNormalizedDiff += Math.pow(normalizedDiff, 2);
            }
            
            // 1. Raw variance
            configResult.rawVariance = sumSquaredDiff / configResult.evaluation.totalFeatures;
            
            // 2. Normalized variance
            configResult.normalizedVariance = sumSquaredNormalizedDiff / configResult.evaluation.totalFeatures;
            
            // 3. Coefficient of variation (CV)
            double stdDev = Math.sqrt(configResult.rawVariance);
            configResult.clusterCountStdDev = stdDev;
            configResult.coefficientOfVariation = (configResult.avgClustersPerFeature > 0) ? 
                (stdDev / configResult.avgClustersPerFeature) : 0.0;
            
            // Set the default variance to coefficient of variation for backward compatibility
            configResult.clusterCountVariance = configResult.coefficientOfVariation;
        }
    }
    
    /**
     * Generate synthetic data with specific configuration (for final output)
     */
    public static Instances generateSyntheticDataWithConfiguration(Instances data, ProfTools profTools, 
                                                                  int maxClusterSize) throws Exception {
        // Determine if there is a class attribute
        int classIndex = data.classIndex();
        boolean hasClassAttribute = classIndex != -1;

        // Save the class attribute if it exists
        Instances originalDataWithClass = null;
        if (hasClassAttribute) {
            originalDataWithClass = new Instances(data);
        }

        // Iterate over each attribute to perform clustering
        for (int featureIndex = 0; featureIndex < data.numAttributes(); featureIndex++) {
            // Skip the class attribute if it exists
            if (featureIndex == classIndex) continue;

            // Create single feature dataset for clustering
            Instances singleFeatureData = createSingleFeatureDataset(data, featureIndex);

            // Perform VWC clustering with optimal parameters
            ClusteringResults clusterResults = performVWCClusteringWithParams(singleFeatureData, maxClusterSize);

            // Replace original feature values with cluster IDs
            int[] clusterAssignments = clusterResults.assignments;
            for (int i = 0; i < data.numInstances(); i++) {
                data.instance(i).setValue(featureIndex, clusterAssignments[i]);
            }
        }

        // Reattach the original class attribute if it exists
        if (hasClassAttribute) {
            for (int i = 0; i < data.numInstances(); i++) {
                data.instance(i).setClassValue(originalDataWithClass.instance(i).classValue());
            }
        }

        return data;
    }


    /**
     * Class to store clustering results
     */
    public static class ClusteringResults {
        public int[] assignments;
        public int numClusters;
        public double totalWithinClusterVariance;
        public Map<Integer, ArrayList<Double>> clusterValues = new HashMap<>();
    }
    
    /**
     * Perform VWC clustering with specified parameters
     */
    public static ClusteringResults performVWCClusteringWithParams(Instances singleFeatureData, 
                                                                  int maxClusterSize) throws Exception {
        // Instantiate the VWC class with the single-feature dataset
        VWC myV = new VWC(singleFeatureData);
        myV.setDistanceFunction(new EuclideanDistance(singleFeatureData));

        // Set parameters - beta = 30% of maxClusterSize (rounded)
        // s_Max=5 -> beta=2, s_Max=10 -> beta=3, s_Max=15 -> beta=5, etc.
        int betaInstances = (int) Math.round(maxClusterSize * 0.30);
        if (betaInstances < 2) betaInstances = 2; // minimum beta = 2
        myV.setBeta(betaInstances);
        myV.setMaxClusterSize(maxClusterSize);

        // Perform the clustering
        myV.buildClusterer(singleFeatureData);

        // Get cluster assignments
        int[] clusterAssignments = myV.getAssignments();
        
        // Calculate clustering statistics
        ClusteringResults results = new ClusteringResults();
        results.assignments = clusterAssignments;
        results.numClusters = myV.numberOfClusters();
        
        // Group values by cluster
        for (int i = 0; i < clusterAssignments.length; i++) {
            int clusterId = clusterAssignments[i];
            double value = singleFeatureData.instance(i).value(0);
            
            results.clusterValues.computeIfAbsent(clusterId, k -> new ArrayList<>()).add(value);
        }
        
        // Calculate total within-cluster variance
        results.totalWithinClusterVariance = 0.0;
        for (Map.Entry<Integer, ArrayList<Double>> entry : results.clusterValues.entrySet()) {
            ArrayList<Double> clusterVals = entry.getValue();
            if (clusterVals.size() > 1) {
                double clusterMean = clusterVals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                double clusterVariance = clusterVals.stream()
                    .mapToDouble(val -> Math.pow(val - clusterMean, 2))
                    .sum() / clusterVals.size();
                results.totalWithinClusterVariance += clusterVals.size() * clusterVariance;
            }
        }
        
        return results;
    }

    
    public static void generateParameterExplorationReport(String datasetName, String outputDirectory) throws IOException {
        String reportFileName = outputDirectory + File.separator + 
                               datasetName.replace(".csv", "_parameter_exploration_report.txt");
        
        List<ConfigurationResult> results = allConfigurationResults.get(datasetName);
        
        try (FileWriter writer = new FileWriter(reportFileName)) {
            writer.write("PARAMETER EXPLORATION REPORT\n");
            writer.write("=".repeat(120) + "\n");
            writer.write("Dataset: " + datasetName + "\n");
            writer.write("Generated: " + new java.util.Date() + "\n");
            double betaPercentage = ConfigLoader.getDoubleProperty("vwc.betaPercentage");
            writer.write("Beta Value: " + String.format("%.1f%% (Fixed)", betaPercentage) + "\n");
            writer.write("Configurations Tested: " + results.size() + "\n");
            writer.write("=".repeat(120) + "\n\n");
            
            // Configuration comparison table with all variance metrics
            writer.write("CONFIGURATION COMPARISON\n");
            writer.write("-".repeat(150) + "\n");
            writer.write(String.format("%-15s %-12s %-12s %-12s %-12s %-15s %-15s %-15s %-15s %-15s%n",
                        "MaxClusterSize", "Utility", "Anonymity", "Combined", "Features", 
                        "AvgClusters", "RawVar", "NormVar", "CV", "ProcessingTime"));
            writer.write("-".repeat(150) + "\n");
            
            for (ConfigurationResult config : results) {
                writer.write(String.format("%-15d %-12.3f %-12.1f %-12.3f %-12d %-15.2f %-15.4f %-15.4f %-15.4f %-15.2fs%n",
                            config.maxClusterSize,
                            config.evaluation.overallUtility,
                            config.evaluation.overallAnonymity,
                            config.evaluation.overallCombined,
                            config.evaluation.totalFeatures,
                            config.avgClustersPerFeature,
                            config.rawVariance,
                            config.normalizedVariance,
                            config.coefficientOfVariation,
                            config.processingTimeMs / 1000.0));
            }
            
            // Find optimal configuration
            ConfigurationResult optimal = results.stream()
                .max((r1, r2) -> Double.compare(r1.evaluation.overallCombined, r2.evaluation.overallCombined))
                .orElse(results.get(0));
            
            writer.write("\nOPTIMAL CONFIGURATION\n");
            writer.write("-".repeat(50) + "\n");
            writer.write("Max Cluster Size: " + optimal.maxClusterSize + "\n");
            writer.write("Beta: " + String.format("%.1f%%", optimal.betaPercentage) + "\n");
            writer.write(String.format("Overall Utility: %.3f%n", optimal.evaluation.overallUtility));
            writer.write(String.format("Overall Anonymity: %.1f%n", optimal.evaluation.overallAnonymity));
            writer.write(String.format("Overall Combined: %.3f%n", optimal.evaluation.overallCombined));
            writer.write(String.format("Average Clusters per Feature: %.2f%n", optimal.avgClustersPerFeature));
            writer.write(String.format("Raw Variance: %.4f%n", optimal.rawVariance));
            writer.write(String.format("Normalized Variance: %.4f%n", optimal.normalizedVariance));
            writer.write(String.format("Coefficient of Variation: %.4f%n", optimal.coefficientOfVariation));
            writer.write(String.format("Processing Time: %.2fs%n", optimal.processingTimeMs / 1000.0));
            
            // Cluster statistics summary
            generateClusterStatisticsSummary(writer, optimal);
            
            // Performance analysis
            writer.write("\nPERFORMANCE ANALYSIS\n");
            writer.write("-".repeat(50) + "\n");
            
            // Best utility
            ConfigurationResult bestUtility = results.stream()
                .max((r1, r2) -> Double.compare(r1.evaluation.overallUtility, r2.evaluation.overallUtility))
                .orElse(results.get(0));
            writer.write("Best Utility: MaxClusterSize=" + bestUtility.maxClusterSize + 
                        " (Utility=" + String.format("%.3f", bestUtility.evaluation.overallUtility) + 
                        ", AvgClusters=" + String.format("%.2f", bestUtility.avgClustersPerFeature) + ")\n");
            
            // Best anonymity
            ConfigurationResult bestAnonymity = results.stream()
                .max((r1, r2) -> Double.compare(r1.evaluation.overallAnonymity, r2.evaluation.overallAnonymity))
                .orElse(results.get(0));
            writer.write("Best Anonymity: MaxClusterSize=" + bestAnonymity.maxClusterSize + 
                        " (Anonymity=" + String.format("%.1f", bestAnonymity.evaluation.overallAnonymity) + 
                        ", AvgClusters=" + String.format("%.2f", bestAnonymity.avgClustersPerFeature) + ")\n");
            
            // Most clusters
            ConfigurationResult mostClusters = results.stream()
                .max((r1, r2) -> Double.compare(r1.avgClustersPerFeature, r2.avgClustersPerFeature))
                .orElse(results.get(0));
            writer.write("Most Clusters: MaxClusterSize=" + mostClusters.maxClusterSize + 
                        " (AvgClusters=" + String.format("%.2f", mostClusters.avgClustersPerFeature) + 
                        ", Utility=" + String.format("%.3f", mostClusters.evaluation.overallUtility) + ")\n");
            
            // Fastest processing
            ConfigurationResult fastest = results.stream()
                .min((r1, r2) -> Long.compare(r1.processingTimeMs, r2.processingTimeMs))
                .orElse(results.get(0));
            writer.write("Fastest Processing: MaxClusterSize=" + fastest.maxClusterSize + 
                        " (" + String.format("%.2fs", fastest.processingTimeMs / 1000.0) + ")\n");
            
            // Trade-off analysis
            writer.write("\nTRADE-OFF ANALYSIS\n");
            writer.write("-".repeat(50) + "\n");
            generateTradeOffAnalysis(writer, results);
            
            // Recommendations
            writer.write("\nRECOMMENDATIONS\n");
            writer.write("-".repeat(50) + "\n");
            generateParameterRecommendations(writer, results, optimal);
        }
        
        System.out.println("Parameter exploration report saved to: " + reportFileName);
    }
    
    /**
     * Generate cluster statistics summary
     */
    public static void generateClusterStatisticsSummary(FileWriter writer, ConfigurationResult config) 
        throws IOException {
        
        writer.write("\nCLUSTER STATISTICS SUMMARY\n");
        writer.write("-".repeat(80) + "\n");
        
        // Cluster count statistics
        writer.write("Cluster Count Statistics:\n");
        writer.write(String.format("  Total Features: %d\n", config.evaluation.totalFeatures));
        writer.write(String.format("  Total Clusters: %d\n", config.totalClusters));
        writer.write(String.format("  Average Clusters per Feature: %.2f\n", config.avgClustersPerFeature));
        writer.write(String.format("  Raw Variance: %.4f\n", config.rawVariance));
        writer.write(String.format("  Normalized Variance: %.4f\n", config.normalizedVariance));
        writer.write(String.format("  Coefficient of Variation: %.4f\n", config.coefficientOfVariation));
        writer.write(String.format("  Standard Deviation: %.4f\n", config.clusterCountStdDev));
        writer.write(String.format("  Min Clusters for any Feature: %d\n", config.minClusters));
        writer.write(String.format("  Max Clusters for any Feature: %d\n", config.maxClusters));
        
        // Count features with only 1 cluster
        int singleClusterFeatures = 0;
        for (Integer count : config.clusterCountsPerFeature.values()) {
            if (count == 1) singleClusterFeatures++;
        }
        
        writer.write(String.format("  Features with only 1 cluster: %d (%.1f%%)\n", 
                                 singleClusterFeatures, 
                                 100.0 * singleClusterFeatures / config.evaluation.totalFeatures));
        
        // Cluster count distribution
        writer.write("\nCluster Count Distribution:\n");
        Map<Integer, Integer> countDistribution = new HashMap<>();
        for (Integer count : config.clusterCountsPerFeature.values()) {
            countDistribution.merge(count, 1, Integer::sum);
        }
        
        for (Map.Entry<Integer, Integer> entry : countDistribution.entrySet()) {
            writer.write(String.format("  %d clusters: %d features (%.1f%%)\n", 
                        entry.getKey(), entry.getValue(), 
                        100.0 * entry.getValue() / config.evaluation.totalFeatures));
        }
    }
    
    /**
     * Generate trade-off analysis
     */
    public static void generateTradeOffAnalysis(FileWriter writer, List<ConfigurationResult> results) throws IOException {
        // Analyze utility vs anonymity trade-off
        writer.write("Utility vs Anonymity Trade-off:\n");
        
        for (ConfigurationResult config : results) {
            String tradeOffType = "";
            if (config.evaluation.overallUtility > 0.8 && config.evaluation.overallAnonymity > 15) {
                tradeOffType = "Excellent Balance";
            } else if (config.evaluation.overallUtility > 0.9) {
                tradeOffType = "Utility-Focused";
            } else if (config.evaluation.overallAnonymity > 20) {
                tradeOffType = "Privacy-Focused";
            } else {
                tradeOffType = "Moderate Performance";
            }
            
            writer.write(String.format("  MaxClusterSize=%d: %s (U=%.3f, A=%.1f, AvgClusters=%.2f, CV=%.4f)%n",
                        config.maxClusterSize, tradeOffType, 
                        config.evaluation.overallUtility, config.evaluation.overallAnonymity,
                        config.avgClustersPerFeature, config.coefficientOfVariation));
        }
        
        // Performance trends
        writer.write("\nPerformance Trends:\n");
        writer.write("  - Smaller cluster sizes generally provide better utility preservation\n");
        writer.write("  - Larger cluster sizes generally provide stronger anonymity protection\n");
        writer.write("  - Average clusters per feature decreases as max cluster size increases\n");
        writer.write("  - Coefficient of variation indicates how uniform the clustering is across features\n");
        writer.write("  - Optimal balance depends on specific privacy-utility requirements\n");
    }
    
    /**
     * Generate parameter recommendations
     */
    public static void generateParameterRecommendations(FileWriter writer, List<ConfigurationResult> results, 
                                                       ConfigurationResult optimal) throws IOException {
        writer.write("Based on the parameter exploration results:\n\n");
        
        writer.write("1. OPTIMAL CONFIGURATION:\n");
        writer.write("   Use MaxClusterSize=" + optimal.maxClusterSize + " for best overall performance\n");
        writer.write("   This provides the best balance of utility and privacy protection\n");
        writer.write("   Average Clusters per Feature: " + String.format("%.2f", optimal.avgClustersPerFeature) + "\n");
        writer.write("   Coefficient of Variation: " + String.format("%.4f", optimal.coefficientOfVariation) + "\n\n");
        
        // Scenario-based recommendations
        writer.write("2. SCENARIO-BASED RECOMMENDATIONS:\n");
        
        // High utility requirement
        ConfigurationResult bestUtility = results.stream()
            .max((r1, r2) -> Double.compare(r1.evaluation.overallUtility, r2.evaluation.overallUtility))
            .orElse(optimal);
        writer.write("   High Utility Requirement: MaxClusterSize=" + bestUtility.maxClusterSize + 
                    " (Utility=" + String.format("%.3f", bestUtility.evaluation.overallUtility) + 
                    ", AvgClusters=" + String.format("%.2f", bestUtility.avgClustersPerFeature) + ")\n");
        
        // High privacy requirement
        ConfigurationResult bestAnonymity = results.stream()
            .max((r1, r2) -> Double.compare(r1.evaluation.overallAnonymity, r2.evaluation.overallAnonymity))
            .orElse(optimal);
        writer.write("   High Privacy Requirement: MaxClusterSize=" + bestAnonymity.maxClusterSize + 
                    " (Anonymity=" + String.format("%.1f", bestAnonymity.evaluation.overallAnonymity) + 
                    ", AvgClusters=" + String.format("%.2f", bestAnonymity.avgClustersPerFeature) + ")\n");
        
        // Performance requirement
        ConfigurationResult fastest = results.stream()
            .min((r1, r2) -> Long.compare(r1.processingTimeMs, r2.processingTimeMs))
            .orElse(optimal);
        writer.write("   Fast Processing Requirement: MaxClusterSize=" + fastest.maxClusterSize + 
                    " (" + String.format("%.2fs", fastest.processingTimeMs / 1000.0) + ")\n\n");
        
        writer.write("3. PARAMETER SENSITIVITY:\n");
        double betaPercentage = ConfigLoader.getDoubleProperty("vwc.betaPercentage");
        writer.write("   Beta value is fixed at " + String.format("%.1f%%", betaPercentage) + 
                    " as requested\n");
        writer.write("   MaxClusterSize is the primary parameter affecting privacy-utility trade-off\n");
        writer.write("   Consider domain-specific requirements when selecting final parameters\n");
    }

    /**
     * Generate final comparison report across all datasets
     */
    public static void generateFinalComparisonReport(String outputDirectory) throws IOException {
        String reportFileName = outputDirectory + File.separator + "final_parameter_comparison_report.txt";
        
        try (FileWriter writer = new FileWriter(reportFileName)) {
            writer.write("FINAL PARAMETER EXPLORATION COMPARISON REPORT\n");
            writer.write("=".repeat(150) + "\n");
            writer.write("Generated: " + new java.util.Date() + "\n");
            double betaPercentage = ConfigLoader.getDoubleProperty("vwc.betaPercentage");
            writer.write("Beta Value: " + String.format("%.1f%% (Fixed for all datasets)", betaPercentage) + "\n");
            writer.write("=".repeat(150) + "\n\n");
            
            // Overall summary table with all variance metrics
            writer.write("OPTIMAL CONFIGURATIONS SUMMARY\n");
            writer.write("-".repeat(150) + "\n");
            writer.write(String.format("%-30s %-15s %-12s %-12s %-12s %-15s %-15s %-15s %-15s %-15s%n",
                        "Dataset", "OptimalClusterSize", "Utility", "Anonymity", "Combined", 
                        "AvgClusters", "RawVar", "NormVar", "CV", "ProcessingTime"));
            writer.write("-".repeat(150) + "\n");
            
            for (Map.Entry<String, List<ConfigurationResult>> entry : allConfigurationResults.entrySet()) {
                String datasetName = entry.getKey();
                List<ConfigurationResult> results = entry.getValue();
                
                ConfigurationResult optimal = results.stream()
                    .max((r1, r2) -> Double.compare(r1.evaluation.overallCombined, r2.evaluation.overallCombined))
                    .orElse(results.get(0));
                
                writer.write(String.format("%-30s %-15d %-12.3f %-12.1f %-12.3f %-15.2f %-15.4f %-15.4f %-15.4f %-15.2fs%n",
                            datasetName, optimal.maxClusterSize,
                            optimal.evaluation.overallUtility,
                            optimal.evaluation.overallAnonymity,
                            optimal.evaluation.overallCombined,
                            optimal.avgClustersPerFeature,
                            optimal.rawVariance,
                            optimal.normalizedVariance,
                            optimal.coefficientOfVariation,
                            optimal.processingTimeMs / 1000.0));
            }
            
            // Cross-dataset analysis
            writer.write("\nCROSS-DATASET ANALYSIS\n");
            writer.write("-".repeat(80) + "\n");
            generateCrossDatasetAnalysis(writer);
            
            // General recommendations
            writer.write("\nGENERAL RECOMMENDATIONS\n");
            writer.write("-".repeat(80) + "\n");
            generateGeneralRecommendations(writer);
        }
        
        System.out.println("Final comparison report saved to: " + reportFileName);
    }
    
    /**
     * Generate cross-dataset analysis
     */
    public static void generateCrossDatasetAnalysis(FileWriter writer) throws IOException {
        // Find most common optimal cluster size
        Map<Integer, Integer> clusterSizeFrequency = new HashMap<>();
        double totalUtility = 0.0, totalAnonymity = 0.0, totalCombined = 0.0;
        double totalAvgClusters = 0.0, totalRawVar = 0.0, totalNormVar = 0.0, totalCV = 0.0;
        int datasetCount = 0;
        
        for (List<ConfigurationResult> results : allConfigurationResults.values()) {
            ConfigurationResult optimal = results.stream()
                .max((r1, r2) -> Double.compare(r1.evaluation.overallCombined, r2.evaluation.overallCombined))
                .orElse(results.get(0));
            
            clusterSizeFrequency.merge(optimal.maxClusterSize, 1, Integer::sum);
            totalUtility += optimal.evaluation.overallUtility;
            totalAnonymity += optimal.evaluation.overallAnonymity;
            totalCombined += optimal.evaluation.overallCombined;
            totalAvgClusters += optimal.avgClustersPerFeature;
            totalRawVar += optimal.rawVariance;
            totalNormVar += optimal.normalizedVariance;
            totalCV += optimal.coefficientOfVariation;
            datasetCount++;
        }
        
        // Most frequent optimal cluster size
        int mostCommonClusterSize = clusterSizeFrequency.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(0);
        
        writer.write("Most Common Optimal Cluster Size: " + mostCommonClusterSize + 
                    " (used by " + clusterSizeFrequency.get(mostCommonClusterSize) + "/" + datasetCount + " datasets)\n");
        writer.write(String.format("Average Performance Across Datasets: Utility=%.3f, Anonymity=%.1f, Combined=%.3f%n",
                    totalUtility / datasetCount, totalAnonymity / datasetCount, totalCombined / datasetCount));
        writer.write(String.format("Average Cluster Statistics: AvgClusters=%.2f, RawVar=%.4f, NormVar=%.4f, CV=%.4f%n",
                    totalAvgClusters / datasetCount, totalRawVar / datasetCount, 
                    totalNormVar / datasetCount, totalCV / datasetCount));
        
        writer.write("\nCluster Size Distribution:\n");
        clusterSizeFrequency.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                try {
                    writer.write(String.format("  MaxClusterSize=%d: %d datasets%n", entry.getKey(), entry.getValue()));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
    }
    
    /**
     * Generate general recommendations
     */
    public static void generateGeneralRecommendations(FileWriter writer) throws IOException {
        writer.write("1. PARAMETER SELECTION STRATEGY:\n");
        writer.write("   - Start with the most common optimal cluster size across datasets\n");
        writer.write("   - Fine-tune based on specific dataset characteristics\n");
        writer.write("   - Consider domain-specific privacy and utility requirements\n\n");
        
        writer.write("2. IMPLEMENTATION GUIDELINES:\n");
        double betaPercentage = ConfigLoader.getDoubleProperty("vwc.betaPercentage");
        writer.write("   - Beta value of " + String.format("%.1f%%", betaPercentage) + 
                    " provides good balance for width parameter estimation\n");
        writer.write("   - Monitor both utility and anonymity metrics during parameter selection\n");
        writer.write("   - Use combined metric for overall optimization guidance\n");
        writer.write("   - Pay attention to average clusters per feature to understand anonymity values\n");
        writer.write("   - Use coefficient of variation (CV) to assess clustering uniformity across features\n\n");
        
        writer.write("3. QUALITY ASSURANCE:\n");
        writer.write("   - Validate results on representative data samples\n");
        writer.write("   - Consider computational efficiency for large-scale deployments\n");
        writer.write("   - Regularly re-evaluate parameters as data characteristics change\n");
        writer.write("   - Monitor cluster count distribution to ensure adequate privacy protection\n");
        writer.write("   - Compare all three variance metrics to gain deeper insights into clustering behavior\n");
    }


    // ==================== METRIC CALCULATION METHODS ====================
    
    /**
     * Calculate utility preservation metric U_f
     */
    public static double calculateUtilityPreservation(double originalVariance, double withinClusterVariance) {
        if (originalVariance == 0.0) {
            return 1.0; // Perfect preservation if no original variance
        }
        return Math.max(0.0, 1.0 - (withinClusterVariance / originalVariance));
    }
    
    /**
     * Calculate anonymity strength metric A_f using weighted average cluster size
     * Formula: A_f = Σ(|C_j|²) / n
     * This represents the expected cluster size for a randomly selected data point.
     */
    public static double calculateAnonymityStrength(ClusteringResults clusterResults, int totalInstances) {
        if (clusterResults.numClusters == 0 || totalInstances == 0) {
            return 0.0;
        }

        // Calculate weighted average: Σ(|C_j|²) / n
        double sumSquaredSizes = 0.0;

        // Iterate through each cluster and sum the squared sizes
        for (Map.Entry<Integer, ArrayList<Double>> entry : clusterResults.clusterValues.entrySet()) {
            int clusterSize = entry.getValue().size();
            sumSquaredSizes += clusterSize * clusterSize;  // |C_j|²
        }

        // Divide by total number of instances
        double weightedAverage = sumSquaredSizes / totalInstances;

        return weightedAverage;
    }
    
    /**
     * Calculate combined privacy-utility metric E
     * Formula: E = α * UPI_f + (1 - α) * (A_f / n)
     */
    public static double calculateCombinedMetric(double utilityPreservation, double anonymityStrength,
                                               double alpha, int totalInstances) {
        // Normalize anonymity component: A_f / n
        // Range: [1/n, 1] where higher values indicate better privacy
        double normalizedAnonymity = anonymityStrength / totalInstances;
        return alpha * utilityPreservation + (1 - alpha) * normalizedAnonymity;
    }
    
    /**
     * Calculate overall metrics across all features
     */
    public static void calculateOverallMetrics(EvaluationResults evaluation) {
        if (evaluation.totalFeatures == 0) {
            evaluation.overallUtility = 0.0;
            evaluation.overallAnonymity = 0.0;
            evaluation.overallCombined = 0.0;
            return;
        }
        
        // Calculate weighted averages
        double totalUtility = 0.0;
        double totalAnonymity = 0.0;
        double totalCombined = 0.0;
        
        for (Map.Entry<Integer, Double> entry : evaluation.utilityPreservation.entrySet()) {
            int featureIndex = entry.getKey();
            totalUtility += entry.getValue();
            totalAnonymity += evaluation.anonymityStrength.get(featureIndex);
            totalCombined += evaluation.combinedMetric.get(featureIndex);
        }
        
        evaluation.overallUtility = totalUtility / evaluation.totalFeatures;
        evaluation.overallAnonymity = totalAnonymity / evaluation.totalFeatures;
        evaluation.overallCombined = totalCombined / evaluation.totalFeatures;
    }
    
    // ==================== UTILITY METHODS ====================
    
    /**
     * Determine alpha value based on feature characteristics
     */
    public static double determineAlphaValue(String featureName) {
        String lowerName = featureName.toLowerCase();
        
        // High privacy features (lower alpha - emphasize privacy)
        if (lowerName.contains("id") || lowerName.contains("ip") || lowerName.contains("address") ||
            lowerName.contains("user") || lowerName.contains("name") || lowerName.contains("location")) {
            return 0.2; // Emphasize privacy
        }
        
        // Medium sensitivity features
        if (lowerName.contains("time") || lowerName.contains("date") || lowerName.contains("port") ||
            lowerName.contains("protocol") || lowerName.contains("service")) {
            return 0.5; // Balanced
        }
        
        // Low sensitivity technical features (higher alpha - emphasize utility)
        return 0.7; // Emphasize utility for technical measurements
    }
    
    /**
     * Extract feature values from dataset
     */
    public static double[] extractFeatureValues(Instances data, int featureIndex) {
        double[] values = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            values[i] = data.instance(i).value(featureIndex);
        }
        return values;
    }
    
    /**
     * Calculate mean of values
     */
    public static double calculateMean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    /**
     * Calculate variance of values
     */
    public static double calculateVariance(double[] values, double mean) {
        double sumSquaredDiffs = 0.0;
        for (double value : values) {
            sumSquaredDiffs += Math.pow(value - mean, 2);
        }
        return sumSquaredDiffs / values.length;
    }
    
    /**
     * Create single feature dataset for clustering
     */
    public static Instances createSingleFeatureDataset(Instances data, int featureIndex) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add((Attribute) data.attribute(featureIndex).copy());

        Instances singleFeatureData = new Instances("SingleFeatureData", attributes, data.numInstances());
        singleFeatureData.setClassIndex(-1);

        for (int i = 0; i < data.numInstances(); i++) {
            double[] values = new double[1];
            values[0] = data.instance(i).value(featureIndex);
            singleFeatureData.add(new DenseInstance(1.0, values));
        }
        
        return singleFeatureData;
    }
    
    // ==================== DATA PROCESSING METHODS ====================

    /**
     * Load dataset from file (supports both CSV and ARFF formats)
     */
    public static Instances loadDataset(String filePath) throws Exception {
        File file = new File(filePath);
        Instances data;

        if (filePath.toLowerCase().endsWith(".arff")) {
            // Load ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setSource(file);
            data = loader.getDataSet();
            System.out.println("Loaded ARFF dataset: " + filePath);
        } else {
            // Load CSV file
            CSVLoader loader = new CSVLoader();
            loader.setSource(file);
            data = loader.getDataSet();
            System.out.println("Loaded CSV dataset: " + filePath);
        }

        // Set class attribute to last attribute if not set
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        System.out.println("Dataset loaded: " + data.numInstances() + " instances, " +
                          data.numAttributes() + " attributes");
        return data;
    }

    /**
     * Normalize the given dataset.
     */
    public static Instances normalizeData(Instances data) throws Exception {
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        return Filter.useFilter(data, normalize);
    }

    /**
     * Save the dataset to a CSV file.
     */
    public static void saveDataset(Instances data, String filePath) throws Exception {
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(filePath));
        saver.writeBatch();
        System.out.println("Saved dataset to " + filePath);
    }

    /**
     * Get the synthetic file name based on the original file path and save directory.
     */
    public static String getSyntheticFileName(String outputDirectory, String originalFilePath, boolean normalized) {
        String suffix = normalized ? "_synthetic_normalized.csv" : "_synthetic.csv";
        String fileName = new File(originalFilePath).getName();
        int dotIndex = fileName.lastIndexOf('.');
        fileName = fileName.substring(0, dotIndex) + suffix;
        return outputDirectory + File.separator + fileName;
    }

    /**
     * Generate LaTeX table for publication
     */
    public static void generateLatexTable(String outputDirectory) throws IOException {
        String latexFileName = outputDirectory + File.separator + "cluster_size_analysis_table.tex";

        // Define the desired order of datasets (matching manuscript)
        String[] datasetOrder = {"bot_loT", "CICIoT2023", "Edge-IIoTset", "MQTTset"};

        try (FileWriter writer = new FileWriter(latexFileName)) {
            // Write LaTeX table header
            writer.write("\\begin{table}[htbp]\n");
            writer.write("  \\centering\n");
            writer.write("  \\caption{Impact of Cluster Size Parameter on VWC Performance Metrics}\n");
            writer.write("  \\label{tab:cluster_size_analysis}\n");
            writer.write("  \\resizebox{1.0\\textwidth}{!}{%\n");
            writer.write("    \\begin{tabular}{lccccccc}\n");
            writer.write("      \\textbf{Dataset} \n");
            writer.write("        & \\textbf{s\\_Max} \n");
            writer.write("        & \\textbf{Average\\_UPI} \n");
            writer.write("        & \\textbf{Average\\_A} \n");
            writer.write("        & \\textbf{Average\\_E} \n");
            writer.write("        & \\textbf{AvgClusters} \n");
            writer.write("        & \\textbf{NormVar} \n");
            writer.write("        & \\textbf{P\\_Time\\,(s)} \\\\\n");
            writer.write("      \\hline\n");

            // Process each dataset in the specified order
            boolean isFirstDataset = true;
            for (String datasetPrefix : datasetOrder) {
                // Find the matching dataset key (may have _train.csv suffix)
                String datasetName = null;
                List<ConfigurationResult> results = null;
                for (Map.Entry<String, List<ConfigurationResult>> entry : allConfigurationResults.entrySet()) {
                    if (entry.getKey().startsWith(datasetPrefix)) {
                        datasetName = entry.getKey();
                        results = entry.getValue();
                        break;
                    }
                }
                if (datasetName == null || results == null) continue;
                
                if (!isFirstDataset) {
                    writer.write("                           \\hline\n");
                }
                
                // Sort results by cluster size
                results.sort((r1, r2) -> Integer.compare(r1.maxClusterSize, r2.maxClusterSize));

                // Clean dataset name for display (remove _train.csv suffix)
                String displayName = datasetPrefix.replace("_", "\\_");

                for (int i = 0; i < results.size(); i++) {
                    ConfigurationResult result = results.get(i);

                    if (i == 0) {
                        // First row for this dataset - no dataset name
                        writer.write(String.format("           &  %2d  & %4.2f & %8.2f & %4.2f & %6.0f & %5.2f & %7.2f \\\\\n",
                            result.maxClusterSize,
                            result.evaluation.overallUtility,
                            result.evaluation.overallAnonymity,
                            result.evaluation.overallCombined,
                            result.avgClustersPerFeature,
                            result.normalizedVariance,
                            result.processingTimeMs / 1000.0));
                    } else if (i == results.size() / 2) {
                        // Middle row - include dataset name with bold formatting
                        writer.write(String.format("                          \\textbf{%s} &  %2d  & %4.2f & %8.2f & %4.2f & %6.0f & %5.2f & %7.2f \\\\\n",
                            displayName,
                            result.maxClusterSize,
                            result.evaluation.overallUtility,
                            result.evaluation.overallAnonymity,
                            result.evaluation.overallCombined,
                            result.avgClustersPerFeature,
                            result.normalizedVariance,
                            result.processingTimeMs / 1000.0));
                    } else {
                        // Other rows - no dataset name
                        writer.write(String.format("                           &  %2d  & %4.2f & %8.2f & %4.2f & %6.0f & %5.2f & %7.2f \\\\\n",
                            result.maxClusterSize,
                            result.evaluation.overallUtility,
                            result.evaluation.overallAnonymity,
                            result.evaluation.overallCombined,
                            result.avgClustersPerFeature,
                            result.normalizedVariance,
                            result.processingTimeMs / 1000.0));
                    }
                }
                isFirstDataset = false;
            }
            
            // Write LaTeX table footer
            writer.write("      \\hline\n");
            writer.write("    \\end{tabular}%\n");
            writer.write("  }\n");
            writer.write("\\end{table}\n");
        }
        
        System.out.println("LaTeX table saved to: " + latexFileName);
    }
}
