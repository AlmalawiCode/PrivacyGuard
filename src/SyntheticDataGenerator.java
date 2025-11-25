package privacyguard;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Scanner;

/**
 * Synthetic Data Generator
 *
 * Generates synthetic datasets using various privacy-preserving methods:
 * 1. PrivacyGuard (VWC)
 * 2. Equal-Width Binning
 * 3. k-means (k=500)
 * 4. k-anonymity (k=5)
 * 5. Laplace DP (ε=1.0)
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class SyntheticDataGenerator {

    private static final String ORIGINAL_DIR = "datasets/Original/";
    private static final String OUTPUT_DIR = "output/1_synthetic_data/";

    // Dataset configurations [Display Name, Base Prefix]
    private static final String[][] DATASETS = {
        {"Bot-IoT", "bot_loT"},
        {"CICIoT2023", "CICIoT2023"},
        {"MQTTset", "MQTTset"},
        {"Edge-IIoTset", "Edge-IIoTset"}
    };

    // Method configurations
    private static final String[] METHOD_NAMES = {
        "PrivacyGuard (VWC)",
        "Equal-Width Binning",
        "k-means (k=500)",
        "k-anonymity (k=5)",
        "Laplace DP (ε=1.0)"
    };

    private static final String[] METHOD_FOLDERS = {
        "PrivacyGuard",
        "Equal_Width_Binning",
        "KMeans",
        "KAnonymity",
        "LaplacianDP"
    };

    /**
     * Main menu for synthetic data generation
     */
    public static void showGenerationMenu(Scanner scanner) {
        boolean generating = true;

        while (generating) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║             Synthetic Data Generator                          ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("[1] Generate Synthetic Data");
            System.out.println("[0] Back to Main Menu");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            switch (choice) {
                case 1:
                    generateSyntheticDataWorkflow(scanner);
                    break;
                case 0:
                    generating = false;
                    System.out.println("\n← Returning to Main Menu...\n");
                    break;
                default:
                    System.out.println("\n✗ Invalid choice. Please try again.\n");
            }
        }
    }

    /**
     * Complete workflow for synthetic data generation
     */
    private static void generateSyntheticDataWorkflow(Scanner scanner) {
        // Step 1: Select dataset
        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Step 1: Select Dataset");
        System.out.println("═══════════════════════════════════════════════════════════════");
        for (int i = 0; i < DATASETS.length; i++) {
            System.out.println("[" + i + "] " + DATASETS[i][0]);
        }
        System.out.print("\nEnter choice (0-" + (DATASETS.length - 1) + "): ");

        int datasetChoice = getIntInput(scanner);
        if (datasetChoice < 0 || datasetChoice >= DATASETS.length) {
            System.out.println("\n✗ Invalid dataset choice.\n");
            return;
        }

        String datasetName = DATASETS[datasetChoice][0];
        String datasetPrefix = DATASETS[datasetChoice][1];

        // Step 2: Select method
        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Step 2: Select Privacy-Preserving Method");
        System.out.println("═══════════════════════════════════════════════════════════════");
        for (int i = 0; i < METHOD_NAMES.length; i++) {
            System.out.println("[" + i + "] " + METHOD_NAMES[i]);
        }
        System.out.println("[" + METHOD_NAMES.length + "] Generate All Methods");
        System.out.print("\nEnter choice (0-" + METHOD_NAMES.length + "): ");

        int methodChoice = getIntInput(scanner);
        if (methodChoice < 0 || methodChoice > METHOD_NAMES.length) {
            System.out.println("\n✗ Invalid method choice.\n");
            return;
        }

        // Step 3: Configure cluster size for PrivacyGuard if selected
        int clusterSize = 5; // Default cluster size (s_max)
        if (methodChoice == 0 || methodChoice == METHOD_NAMES.length) {
            System.out.println("\n═══════════════════════════════════════════════════════════════");
            System.out.println("  Step 3: Configure PrivacyGuard s_max (Maximum Cluster Size)");
            System.out.println("═══════════════════════════════════════════════════════════════");
            System.out.println("[1] Use default s_max = 5");
            System.out.println("[2] Enter custom s_max value");
            System.out.print("\nEnter choice (1-2): ");

            int configChoice = getIntInput(scanner);
            if (configChoice == 2) {
                System.out.print("Enter s_max value: ");
                int inputClusterSize = getIntInput(scanner);
                if (inputClusterSize > 0) {
                    clusterSize = inputClusterSize;
                }
            }
            System.out.println("  ✓ s_max set to: " + clusterSize);
        }

        // Generate synthetic data
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║                 Generation in Progress...                     ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        int methodsToGenerate = (methodChoice == METHOD_NAMES.length) ? METHOD_NAMES.length : 1;
        int startMethod = (methodChoice == METHOD_NAMES.length) ? 0 : methodChoice;
        int endMethod = startMethod + methodsToGenerate;

        int successCount = 0;
        int failureCount = 0;

        for (int m = startMethod; m < endMethod; m++) {
            String methodName = METHOD_NAMES[m];
            String methodFolder = METHOD_FOLDERS[m];

            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("  Method: " + methodName);
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Generate synthetic data from complete merged dataset
            String inputFile = ORIGINAL_DIR + datasetPrefix + ".arff";
            String outputFile = OUTPUT_DIR + methodFolder + "/" + datasetPrefix + "_synthetic.arff";

            System.out.println("  Processing complete dataset...");
            GenerationInfo genInfo = generateSynthetic(inputFile, outputFile, m, datasetName, clusterSize);
            if (genInfo != null) {
                successCount++;
                System.out.println("  ✓ Complete");
            } else {
                failureCount++;
                System.out.println("  ✗ Failed");
            }

            // Save summary file
            if (genInfo != null) {
                saveSummaryFile(datasetPrefix, methodName, m, genInfo, clusterSize);
            }

            System.out.println();
        }

        // Summary
        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║                  Generation Summary                           ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println("  Dataset: " + datasetName);
        System.out.println("  Successful: " + successCount);
        System.out.println("  Failed: " + failureCount);
        System.out.println();
    }

    /**
     * Generate synthetic data using specified method
     */
    private static GenerationInfo generateSynthetic(String inputFile, String outputFile, int methodIndex, String splitType, int clusterSize) {
        try {
            long startTime = System.currentTimeMillis();

            // Load original data
            Instances originalData = loadDataset(inputFile);
            originalData.setClassIndex(originalData.numAttributes() - 1);

            Instances syntheticData = null;

            // Generate synthetic data based on method
            switch (methodIndex) {
                case 0: // PrivacyGuard (VWC)
                    PrivacyGuardGenerator privacyGuard = new PrivacyGuardGenerator(clusterSize, 0.5);
                    syntheticData = privacyGuard.generateSyntheticData(originalData);
                    break;

                case 1: // Equal-Width Binning
                    EqualWidthBinning binning = new EqualWidthBinning(10);
                    syntheticData = binning.generateSyntheticData(originalData);
                    break;

                case 2: // k-means
                    KMeansBaseline kmeans = new KMeansBaseline(500);
                    syntheticData = kmeans.generateSyntheticData(originalData);
                    break;

                case 3: // k-anonymity
                    KAnonymity kanonymity = new KAnonymity(5);
                    syntheticData = kanonymity.generateSyntheticData(originalData);
                    break;

                case 4: // Laplace DP
                    LaplaceDPBaseline laplaceDP = new LaplaceDPBaseline(1.0);
                    syntheticData = laplaceDP.generateSyntheticData(originalData);
                    break;

                default:
                    System.err.println("    ✗ Unknown method index: " + methodIndex);
                    return null;
            }

            // Normalize synthetic data to [0, 1] range
            System.out.println("  Normalizing synthetic data to [0, 1] range...");
            syntheticData = normalizeSyntheticData(syntheticData);

            // Save synthetic data
            saveDataset(syntheticData, outputFile);

            long endTime = System.currentTimeMillis();
            double elapsedSeconds = (endTime - startTime) / 1000.0;

            // Calculate additional info (e.g., cluster counts for PrivacyGuard)
            String additionalInfo = "";
            if (methodIndex == 0) { // PrivacyGuard
                additionalInfo = calculateClusterInfo(syntheticData);
            }

            System.out.println("    Instances: " + syntheticData.numInstances() +
                             " | Time: " + String.format("%.2f", elapsedSeconds) + "s");
            System.out.println("    Saved: " + outputFile);

            // Return generation info
            int numFeatures = syntheticData.numAttributes() - 1; // Exclude class attribute
            return new GenerationInfo(splitType, syntheticData.numInstances(),
                                    syntheticData.numClasses(), numFeatures,
                                    elapsedSeconds, outputFile, additionalInfo);

        } catch (Exception e) {
            System.err.println("    ✗ Error: " + e.getMessage());
            return null;
        }
    }

    /**
     * Calculate cluster statistics for PrivacyGuard (VWC) method
     */
    private static String calculateClusterInfo(Instances data) {
        try {
            int classIndex = data.classIndex();
            int totalClusters = 0;
            int minClusters = Integer.MAX_VALUE;
            int maxClusters = 0;

            // For each feature (excluding class)
            for (int i = 0; i < data.numAttributes(); i++) {
                if (i == classIndex) continue;

                // Count unique cluster IDs for this feature
                java.util.Set<Double> uniqueValues = new java.util.HashSet<>();
                for (int j = 0; j < data.numInstances(); j++) {
                    uniqueValues.add(data.instance(j).value(i));
                }

                int numClusters = uniqueValues.size();
                totalClusters += numClusters;
                minClusters = Math.min(minClusters, numClusters);
                maxClusters = Math.max(maxClusters, numClusters);
            }

            int numFeatures = data.numAttributes() - 1;
            double avgClusters = (double) totalClusters / numFeatures;

            return String.format("Total Clusters: %d, Avg per Feature: %.2f, Min: %d, Max: %d",
                               totalClusters, avgClusters, minClusters, maxClusters);

        } catch (Exception e) {
            return "Cluster info unavailable";
        }
    }

    /**
     * Load dataset from file (auto-detects format)
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    /**
     * Save dataset to ARFF file
     */
    private static void saveDataset(Instances data, String filePath) throws Exception {
        // Ensure output directory exists
        File outputFile = new File(filePath);
        outputFile.getParentFile().mkdirs();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(outputFile);
        saver.writeBatch();
    }

    /**
     * Normalize synthetic data to [0, 1] range
     * Applies Weka's Normalize filter to all numeric attributes (except class)
     */
    private static Instances normalizeSyntheticData(Instances data) throws Exception {
        // Create a copy to avoid modifying original
        Instances normalizedData = new Instances(data);

        // Create and configure normalize filter
        Normalize normalize = new Normalize();
        normalize.setInputFormat(normalizedData);

        // Apply normalization (scales to [0, 1])
        normalizedData = Filter.useFilter(normalizedData, normalize);

        return normalizedData;
    }

    /**
     * Get integer input from user (with error handling)
     */
    private static int getIntInput(Scanner scanner) {
        try {
            return scanner.nextInt();
        } catch (Exception e) {
            scanner.nextLine(); // Clear buffer
            return -1; // Invalid input
        }
    }

    /**
     * Helper class to store generation metadata
     */
    private static class GenerationInfo {
        String splitType;
        int numInstances;
        int numClasses;
        int numFeatures;
        double timeSeconds;
        String outputFile;
        String additionalInfo; // For method-specific info like cluster counts

        GenerationInfo(String splitType, int numInstances, int numClasses, int numFeatures,
                      double timeSeconds, String outputFile, String additionalInfo) {
            this.splitType = splitType;
            this.numInstances = numInstances;
            this.numClasses = numClasses;
            this.numFeatures = numFeatures;
            this.timeSeconds = timeSeconds;
            this.outputFile = outputFile;
            this.additionalInfo = additionalInfo;
        }
    }

    /**
     * Get method configuration as string
     */
    private static String getMethodConfig(int methodIndex, int clusterSize) {
        switch (methodIndex) {
            case 0: return "maxClusterSize=" + clusterSize + ", betaPercentage=0.5%";
            case 1: return "numBins=10";
            case 2: return "k=500";
            case 3: return "k=5";
            case 4: return "epsilon=1.0";
            default: return "unknown";
        }
    }

    /**
     * Save generation summary/configuration file
     */
    private static void saveSummaryFile(String datasetName, String methodName, int methodIndex,
                                      GenerationInfo genInfo, int clusterSize) {
        try {
            String methodFolder = METHOD_FOLDERS[methodIndex];
            String summaryFile = OUTPUT_DIR + methodFolder + "/" + datasetName + "_generation_summary.txt";

            File file = new File(summaryFile);
            file.getParentFile().mkdirs();

            PrintWriter writer = new PrintWriter(new FileWriter(file));

            // Header
            writer.println("╔════════════════════════════════════════════════════════════════╗");
            writer.println("║           Synthetic Data Generation Summary                   ║");
            writer.println("╚════════════════════════════════════════════════════════════════╝");
            writer.println();

            // Generation timestamp
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            writer.println("Generation Date: " + sdf.format(new Date()));
            writer.println();

            // Dataset info
            writer.println("Dataset: " + datasetName);
            writer.println("Method: " + methodName);
            writer.println();

            // Method configuration
            writer.println("═══════════════════════════════════════════════════════════════");
            writer.println(" Method Configuration");
            writer.println("═══════════════════════════════════════════════════════════════");
            writer.println("Parameters: " + getMethodConfig(methodIndex, clusterSize));
            writer.println();

            // Synthetic data info
            writer.println("═══════════════════════════════════════════════════════════════");
            writer.println(" Synthetic Data Generated");
            writer.println("═══════════════════════════════════════════════════════════════");
            writer.println("Number of Instances: " + genInfo.numInstances);
            writer.println("Number of Features: " + genInfo.numFeatures);
            writer.println("Number of Classes: " + genInfo.numClasses);
            writer.println("Generation Time: " + String.format("%.2f", genInfo.timeSeconds) + " seconds");
            if (genInfo.additionalInfo != null && !genInfo.additionalInfo.isEmpty()) {
                writer.println("Cluster Info: " + genInfo.additionalInfo);
            }
            writer.println("Output File: " + genInfo.outputFile);
            writer.println();

            writer.println("═══════════════════════════════════════════════════════════════");
            writer.println("Total Generation Time: " + String.format("%.2f", genInfo.timeSeconds) + " seconds");
            writer.println("═══════════════════════════════════════════════════════════════");

            writer.close();

            System.out.println("  ✓ Summary saved: " + summaryFile);

        } catch (Exception e) {
            System.err.println("  ✗ Failed to save summary: " + e.getMessage());
        }
    }
}
