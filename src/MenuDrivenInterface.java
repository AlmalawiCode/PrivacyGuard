package privacyguard;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Main Menu-Driven Interface
 * Central control class for accessing all system functionalities
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class MenuDrivenInterface {

    private static final String DATASET_DIR = "datasets/Original/";
    private static Scanner scanner;

    public static void main(String[] args) {
        scanner = new Scanner(System.in);
        boolean running = true;

        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║     Privacy-Preserving Methods Evaluation System             ║");
        System.out.println("║     Menu-Driven Interface                                     ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        while (running) {
            showMainMenu();
            int choice = getIntInput(scanner);

            switch (choice) {
                case 1:
                    // Generate synthetic data
                    SyntheticDataGenerator.showGenerationMenu(scanner);
                    break;
                case 2:
                    // Generate Train/Test Indices
                    IndicesGenerator.showIndicesMenu(scanner);
                    break;
                case 3:
                    // Classification Evaluation
                    runClassificationEvaluation();
                    break;
                case 4:
                    // Generate LaTeX Tables
                    generateLatexTables();
                    break;
                case 5:
                    // Privacy Attack Evaluation
                    PrivacyAttackEvaluator.showPrivacyAttackMenu(scanner);
                    break;
                case 6:
                    // Parameter Exploration
                    SyntheticDataGenerationWithParameterExploration.showParameterExplorationMenu(scanner);
                    break;
                case 7:
                    // Explore datasets
                    exploreDatasets();
                    break;
                case 0:
                    System.out.println("\n✓ Exiting system. Goodbye!");
                    running = false;
                    break;
                default:
                    System.out.println("\n✗ Invalid choice. Please try again.\n");
            }
        }

        scanner.close();
    }

    /**
     * Display main menu
     */
    private static void showMainMenu() {
        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║                      Main Menu                                ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println("[1] Generate Synthetic Data - Create privacy-preserving datasets");
        System.out.println("[2] Generate Train/Test Indices - Create consistent train/test splits");
        System.out.println("[3] Classification Evaluation - Evaluate classifiers on all methods");
        System.out.println("[4] Generate LaTeX Tables - Convert results to LaTeX format");
        System.out.println("[5] Privacy Attack Evaluation - Test re-identification & linkage attacks");
        System.out.println("[6] Parameter Exploration - Analyze impact of s_max on utility/privacy");
        System.out.println("[7] Explore Datasets - View dataset details and distributions");
        System.out.println("[0] Exit");
        System.out.print("\nEnter choice: ");
    }

    /**
     * Explore datasets - integrates with dataset exploration functionality
     */
    private static void exploreDatasets() {
        boolean exploring = true;

        while (exploring) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║                   Dataset Explorer                            ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("[1] Show Dataset Details");
            System.out.println("[0] Back to Main Menu");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            switch (choice) {
                case 1:
                    showDatasetDetails();
                    break;
                case 0:
                    exploring = false;
                    System.out.println("\n← Returning to Main Menu...\n");
                    break;
                default:
                    System.out.println("\n✗ Invalid choice. Please try again.\n");
            }
        }
    }

    /**
     * Show dataset selection menu and display details
     */
    private static void showDatasetDetails() {
        // Dynamically scan for ARFF and CSV files in the directory
        File dir = new File(DATASET_DIR);
        File[] files = dir.listFiles((d, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".arff") || lower.endsWith(".csv");
        });

        if (files == null || files.length == 0) {
            System.out.println("\n✗ No datasets found in " + DATASET_DIR + "\n");
            return;
        }

        // Sort files alphabetically
        Arrays.sort(files);

        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Select Dataset");
        System.out.println("═══════════════════════════════════════════════════════════════");
        System.out.println("  Found " + files.length + " datasets in " + DATASET_DIR);
        System.out.println();

        for (int i = 0; i < files.length; i++) {
            System.out.println("[" + i + "] " + files[i].getName());
        }
        System.out.println("[" + files.length + "] Back");
        System.out.print("\nEnter choice (0-" + files.length + "): ");

        int choice = getIntInput(scanner);

        if (choice >= 0 && choice < files.length) {
            String fileName = files[choice].getName();
            String filePath = DATASET_DIR + fileName;
            displayDatasetInfo(fileName, filePath);
        } else if (choice == files.length) {
            System.out.println("\n← Going back...\n");
        } else {
            System.out.println("\n✗ Invalid dataset choice.\n");
        }
    }

    /**
     * Load and display dataset information
     */
    private static void displayDatasetInfo(String datasetName, String filePath) {
        try {
            System.out.println("\n⏳ Loading dataset...");

            // Load dataset
            Instances data = loadDataset(filePath);
            data.setClassIndex(data.numAttributes() - 1);

            // Display header
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║  Dataset: " + padRight(datasetName, 52) + "║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println();

            // Display basic statistics
            System.out.println("📊 Basic Statistics:");
            System.out.println("  Total Instances     : " + String.format("%,d", data.numInstances()));
            System.out.println("  Number of Features  : " + (data.numAttributes() - 1) + " (excluding class)");
            System.out.println("  Number of Classes   : " + data.numClasses());
            System.out.println();

            // Display class distribution
            System.out.println("📈 Class Distribution:");
            System.out.println("  " + padRight("Class Label", 20) + " | " + padLeft("Count", 10) + " | " + padLeft("Percentage", 12));
            System.out.println("  " + repeat("-", 20) + "-+-" + repeat("-", 10) + "-+-" + repeat("-", 12));

            // Count instances per class
            int[] classCounts = new int[data.numClasses()];
            for (int i = 0; i < data.numInstances(); i++) {
                int classValue = (int) data.instance(i).classValue();
                classCounts[classValue]++;
            }

            // Display each class
            for (int i = 0; i < data.numClasses(); i++) {
                String className = data.classAttribute().value(i);
                int count = classCounts[i];
                double percentage = (count * 100.0) / data.numInstances();

                System.out.println("  " + padRight(className, 20) + " | " +
                                 padLeft(String.format("%,d", count), 10) + " | " +
                                 padLeft(String.format("%.2f%%", percentage), 12));
            }

            System.out.println();
            System.out.println("✓ Dataset loaded successfully!");
            System.out.println();

        } catch (IOException e) {
            System.err.println("\n✗ Error loading dataset: " + e.getMessage());
            System.err.println("  File: " + filePath);
            System.out.println();
        } catch (Exception e) {
            System.err.println("\n✗ Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.out.println();
        }
    }

    /**
     * Load dataset from file (supports CSV, ARFF, etc.)
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
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

    /**
     * Run classification evaluation
     */
    private static void runClassificationEvaluation() {
        boolean evaluating = true;

        while (evaluating) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║              Classification Evaluation                       ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("[1] Evaluate bot_loT");
            System.out.println("[2] Evaluate CICIoT2023");
            System.out.println("[3] Evaluate MQTTset");
            System.out.println("[4] Evaluate Edge-IIoTset");
            System.out.println("[5] Evaluate All Datasets");
            System.out.println("[0] Back to Main Menu");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            try {
                switch (choice) {
                    case 1:
                        ClassificationEvaluator.evaluateDataset("bot_loT");
                        break;
                    case 2:
                        ClassificationEvaluator.evaluateDataset("CICIoT2023");
                        break;
                    case 3:
                        ClassificationEvaluator.evaluateDataset("MQTTset");
                        break;
                    case 4:
                        ClassificationEvaluator.evaluateDataset("Edge-IIoTset");
                        break;
                    case 5:
                        System.out.println("\n⚠ Evaluating all datasets. This may take a long time...\n");
                        ClassificationEvaluator.evaluateDataset("bot_loT");
                        ClassificationEvaluator.evaluateDataset("CICIoT2023");
                        ClassificationEvaluator.evaluateDataset("MQTTset");
                        ClassificationEvaluator.evaluateDataset("Edge-IIoTset");
                        System.out.println("\n✓ All datasets evaluated successfully!\n");
                        break;
                    case 0:
                        evaluating = false;
                        System.out.println("\n← Returning to Main Menu...\n");
                        break;
                    default:
                        System.out.println("\n✗ Invalid choice. Please try again.\n");
                }
            } catch (Exception e) {
                System.err.println("\n✗ Error during evaluation: " + e.getMessage());
                e.printStackTrace();
                System.out.println();
            }
        }
    }

    /**
     * Generate LaTeX tables from evaluation results
     */
    private static void generateLatexTables() {
        boolean generating = true;

        while (generating) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║              Generate LaTeX Tables                           ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("[1] Generate Classification Results Tables");
            System.out.println("[2] Generate Privacy Attack Results Tables");
            System.out.println("[3] Generate ALL Summary Tables (4 Options)");
            System.out.println("[0] Back to Main Menu");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            try {
                switch (choice) {
                    case 1:
                        LaTeXTableGenerator.generateAllTablesInOneFile();
                        break;
                    case 2:
                        LaTeXTableGenerator.generatePrivacyResultsTables();
                        break;
                    case 3:
                        LaTeXTableGenerator.generateAllSummaryTables();
                        break;
                    case 0:
                        generating = false;
                        System.out.println("\n← Returning to Main Menu...\n");
                        break;
                    default:
                        System.out.println("\n✗ Invalid choice. Please try again.\n");
                }
            } catch (Exception e) {
                System.err.println("\n✗ Error during LaTeX generation: " + e.getMessage());
                e.printStackTrace();
                System.out.println();
            }
        }
    }
}
