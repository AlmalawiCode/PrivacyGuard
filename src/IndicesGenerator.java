package privacyguard;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.*;
import java.util.*;

/**
 * Generate Training and Testing Indices for Fair Evaluation
 *
 * Creates consistent train/test splits that:
 * - Use stratified sampling (70/30 split)
 * - Ensure minimum 5 instances per class in testing
 * - Handle small classes (< 10 instances) by including all in both sets
 * - Use 0-based indexing (Weka standard)
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class IndicesGenerator {

    // Configuration constants
    private static final int RANDOM_SEED = 180;  // Change this for different splits
    private static final double TRAIN_RATIO = 0.7;  // 70% training, 30% testing
    private static final int MIN_TEST_PER_CLASS = 5;  // Minimum instances per class in test
    private static final int MIN_CLASS_SIZE_FOR_DUPLICATION = 10;  // If < 10, put all in both sets

    private static final String ORIGINAL_DIR = "datasets/Original/";
    private static final String OUTPUT_DIR = "output/2_indices/";
    private static final String[][] DATASETS = {
        {"Bot-IoT", "bot_loT"},
        {"CICIoT2023", "CICIoT2023"},
        {"MQTTset", "MQTTset"},
        {"Edge-IIoTset", "Edge-IIoTset"}
    };

    /**
     * Main menu for indices generation
     */
    public static void showIndicesMenu(Scanner scanner) {
        boolean generating = true;

        while (generating) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║          Train/Test Indices Generator                        ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("Generate consistent train/test splits for fair evaluation");
            System.out.println("Configuration:");
            System.out.println("  • Train/Test Split: " + (int)(TRAIN_RATIO * 100) + "/" + (int)((1 - TRAIN_RATIO) * 100));
            System.out.println("  • Random Seed: " + RANDOM_SEED);
            System.out.println("  • Min Test per Class: " + MIN_TEST_PER_CLASS);
            System.out.println("  • Small Class Threshold: " + MIN_CLASS_SIZE_FOR_DUPLICATION);
            System.out.println();
            System.out.println("[1] Generate Indices for Specific Dataset");
            System.out.println("[2] Generate Indices for All Datasets");
            System.out.println("[0] Back to Main Menu");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            switch (choice) {
                case 1:
                    generateForSpecificDataset(scanner);
                    break;
                case 2:
                    generateForAllDatasets();
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
     * Generate indices for a specific dataset
     */
    private static void generateForSpecificDataset(Scanner scanner) {
        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Select Dataset");
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

        generateIndices(datasetName, datasetPrefix);
    }

    /**
     * Generate indices for all datasets
     */
    private static void generateForAllDatasets() {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║          Generating Indices for All Datasets                 ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        int successCount = 0;
        int failureCount = 0;

        for (String[] dataset : DATASETS) {
            String datasetName = dataset[0];
            String datasetPrefix = dataset[1];

            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("  Dataset: " + datasetName);
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            if (generateIndices(datasetName, datasetPrefix)) {
                successCount++;
            } else {
                failureCount++;
            }

            System.out.println();
        }

        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║                  Generation Summary                           ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println("  Successful: " + successCount);
        System.out.println("  Failed: " + failureCount);
        System.out.println();
    }

    /**
     * Generate training and testing indices for a dataset
     */
    private static boolean generateIndices(String datasetName, String datasetPrefix) {
        try {
            // Load dataset
            String datasetPath = ORIGINAL_DIR + datasetPrefix + ".arff";
            System.out.println("  Loading: " + datasetPath);

            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            System.out.println("  Total instances: " + data.numInstances());
            System.out.println("  Number of classes: " + data.numClasses());

            // Group instances by class
            Map<Integer, List<Integer>> classToInstances = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                int classValue = (int) data.instance(i).classValue();
                classToInstances.computeIfAbsent(classValue, k -> new ArrayList<>()).add(i);
            }

            // Create random number generator with fixed seed
            Random random = new Random(RANDOM_SEED);

            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> testIndices = new ArrayList<>();

            System.out.println("\n  Splitting by class:");

            // Process each class
            for (int classIdx = 0; classIdx < data.numClasses(); classIdx++) {
                List<Integer> classInstances = classToInstances.getOrDefault(classIdx, new ArrayList<>());
                String className = data.classAttribute().value(classIdx);
                int classSize = classInstances.size();

                // Shuffle instances for this class
                Collections.shuffle(classInstances, random);

                // Check if class is too small
                if (classSize < MIN_CLASS_SIZE_FOR_DUPLICATION) {
                    // Put all instances in BOTH training and testing
                    trainIndices.addAll(classInstances);
                    testIndices.addAll(classInstances);
                    System.out.println("    " + padRight(className, 25) + ": " +
                                     padLeft(String.valueOf(classSize), 5) + " instances → " +
                                     "ALL in both train & test (class too small)");
                } else {
                    // Calculate split sizes
                    int numTest = Math.max(MIN_TEST_PER_CLASS, (int) Math.round(classSize * (1 - TRAIN_RATIO)));
                    int numTrain = classSize - numTest;

                    // Ensure we don't exceed available instances
                    if (numTest + numTrain > classSize) {
                        numTest = Math.max(MIN_TEST_PER_CLASS, classSize - numTrain);
                        numTrain = classSize - numTest;
                    }

                    // Split the instances
                    List<Integer> testSet = classInstances.subList(0, numTest);
                    List<Integer> trainSet = classInstances.subList(numTest, classSize);

                    trainIndices.addAll(trainSet);
                    testIndices.addAll(testSet);

                    System.out.println("    " + padRight(className, 25) + ": " +
                                     padLeft(String.valueOf(classSize), 5) + " instances → " +
                                     "train: " + numTrain + ", test: " + numTest);
                }
            }

            // Sort indices for consistent ordering
            Collections.sort(trainIndices);
            Collections.sort(testIndices);

            // Save indices to files
            new File(OUTPUT_DIR).mkdirs();
            String trainFile = OUTPUT_DIR + datasetPrefix + "_indices_training.txt";
            String testFile = OUTPUT_DIR + datasetPrefix + "_indices_testing.txt";

            saveIndices(trainIndices, trainFile);
            saveIndices(testIndices, testFile);

            System.out.println("\n  ✓ Indices generated successfully!");
            System.out.println("    Training indices: " + trainIndices.size() + " → " + trainFile);
            System.out.println("    Testing indices: " + testIndices.size() + " → " + testFile);

            return true;

        } catch (Exception e) {
            System.err.println("  ✗ Error: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Save indices to file (one index per line)
     */
    private static void saveIndices(List<Integer> indices, String filePath) throws IOException {
        PrintWriter writer = new PrintWriter(new FileWriter(filePath));
        for (int index : indices) {
            writer.println(index);
        }
        writer.close();
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
     * Pad string to the right with spaces
     */
    private static String padRight(String str, int length) {
        if (str.length() >= length) {
            return str.substring(0, length);
        }
        return str + repeat(" ", length - str.length());
    }

    /**
     * Pad string to the left with spaces
     */
    private static String padLeft(String str, int length) {
        if (str.length() >= length) {
            return str.substring(0, length);
        }
        return repeat(" ", length - str.length()) + str;
    }

    /**
     * Repeat a string n times
     */
    private static String repeat(String str, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}
