package privacyguard;

import weka.core.Instances;
import weka.core.EuclideanDistance;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Privacy Attack Evaluator - Menu Integration
 *
 * Evaluates privacy guarantees of synthetic data by running:
 * 1. Re-identification attacks
 * 2. Linkage attacks
 *
 * Integrated with the menu system for easy access.
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class PrivacyAttackEvaluator {

    private static final String ORIGINAL_DIR = "datasets/Original/";
    private static final String SYNTHETIC_DIR = "output/1_synthetic_data/";
    private static final String OUTPUT_DIR = "output/5_privacy_attacks/";

    private static final String[][] DATASETS = {
        {"Bot-IoT", "bot_loT"},
        {"CICIoT2023", "CICIoT2023"},
        {"MQTTset", "MQTTset"},
        {"Edge-IIoTset", "Edge-IIoTset"}
    };

    private static final String[] METHODS = {
        "PrivacyGuard",
        "Equal_Width_Binning",
        "KMeans",
        "KAnonymity",
        "LaplacianDP"
    };

    /**
     * Show privacy attack evaluation menu
     */
    public static void showPrivacyAttackMenu(Scanner scanner) {
        boolean evaluating = true;

        while (evaluating) {
            System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
            System.out.println("║          Privacy Attack Evaluation                           ║");
            System.out.println("╚═══════════════════════════════════════════════════════════════╝");
            System.out.println("Test synthetic data privacy using:");
            System.out.println("  • Re-identification Attack (distance-based matching)");
            System.out.println("  • Linkage Attack (partial knowledge simulation)");
            System.out.println();
            System.out.println("[1] Evaluate Specific Dataset & Method");
            System.out.println("[2] Evaluate All Methods for Specific Dataset (PARALLEL - uses 5 cores)");
            System.out.println("[3] Evaluate All Datasets & All Methods");
            System.out.println("[0] Back to Main Menu");
            System.out.println();
            System.out.println("Note: Option [2] runs methods in parallel using 5 cores.");
            System.out.println("      You can run 4 instances with different datasets to use all 20 cores.");
            System.out.print("\nEnter choice: ");

            int choice = getIntInput(scanner);

            try {
                switch (choice) {
                    case 1:
                        evaluateSpecificDatasetAndMethod(scanner);
                        break;
                    case 2:
                        evaluateAllMethodsForDataset(scanner);
                        break;
                    case 3:
                        evaluateAllDatasetsAndMethods();
                        break;
                    case 0:
                        evaluating = false;
                        System.out.println("\n← Returning to Main Menu...\n");
                        break;
                    default:
                        System.out.println("\n✗ Invalid choice. Please try again.\n");
                }
            } catch (Exception e) {
                System.err.println("\n✗ Error during privacy evaluation: " + e.getMessage());
                e.printStackTrace();
                System.out.println();
            }
        }
    }

    /**
     * Evaluate specific dataset and method
     */
    private static void evaluateSpecificDatasetAndMethod(Scanner scanner) throws Exception {
        // Select dataset
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

        // Select method
        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Select Privacy-Preserving Method");
        System.out.println("═══════════════════════════════════════════════════════════════");
        for (int i = 0; i < METHODS.length; i++) {
            System.out.println("[" + i + "] " + METHODS[i]);
        }
        System.out.print("\nEnter choice (0-" + (METHODS.length - 1) + "): ");

        int methodChoice = getIntInput(scanner);
        if (methodChoice < 0 || methodChoice >= METHODS.length) {
            System.out.println("\n✗ Invalid method choice.\n");
            return;
        }

        String datasetName = DATASETS[datasetChoice][0];
        String datasetPrefix = DATASETS[datasetChoice][1];
        String methodName = METHODS[methodChoice];

        evaluateDatasetMethod(datasetName, datasetPrefix, methodName);
    }

    /**
     * Evaluate all methods for a specific dataset
     * OPTIMIZED: Builds BallTree once and shares it across all methods to save memory and time
     */
    private static void evaluateAllMethodsForDataset(Scanner scanner) throws Exception {
        // Select dataset
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

        System.out.println("\n⚠ Evaluating all methods for " + datasetName + " in PARALLEL...\n");
        System.out.println("Using " + METHODS.length + " cores (one per method)\n");

        // Create thread pool with one thread per method
        ExecutorService executor = Executors.newFixedThreadPool(METHODS.length);
        List<Future<?>> futures = new ArrayList<>();
        AtomicInteger completedCount = new AtomicInteger(0);

        // Submit each method evaluation as a separate task
        for (String methodName : METHODS) {
            final String method = methodName;
            Future<?> future = executor.submit(() -> {
                try {
                    evaluateDatasetMethod(datasetName, datasetPrefix, method);
                    int completed = completedCount.incrementAndGet();
                    System.out.println("\n[" + completed + "/" + METHODS.length + "] ✓ Completed: " + method + "\n");
                } catch (Exception e) {
                    System.err.println("\n✗ Error evaluating " + method + ": " + e.getMessage());
                    e.printStackTrace();
                }
            });
            futures.add(future);
        }

        // Wait for all tasks to complete
        System.out.println("Waiting for all methods to complete...\n");
        for (Future<?> future : futures) {
            try {
                future.get(); // Wait for completion
            } catch (Exception e) {
                System.err.println("Error waiting for task: " + e.getMessage());
            }
        }

        // Shutdown executor
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        System.out.println("\n✓ All " + METHODS.length + " methods evaluated for " + datasetName + "!\n");
    }

    /**
     * Evaluate all datasets and all methods
     */
    private static void evaluateAllDatasetsAndMethods() throws Exception {
        System.out.println("\n⚠ Evaluating ALL datasets and ALL methods. This will take time...\n");

        for (String[] dataset : DATASETS) {
            String datasetName = dataset[0];
            String datasetPrefix = dataset[1];

            for (String methodName : METHODS) {
                evaluateDatasetMethod(datasetName, datasetPrefix, methodName);
                System.out.println();
            }
        }

        System.out.println("✓ All datasets and methods evaluated!\n");
    }

    /**
     * Evaluate a specific dataset and method combination
     * @param datasetName Display name of dataset
     * @param datasetPrefix File prefix of dataset
     * @param methodName Name of synthetic data generation method
     */
    public static void evaluateDatasetMethod(String datasetName, String datasetPrefix, String methodName) throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║  Privacy Attack Evaluation                                   ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println("Dataset: " + datasetName);
        System.out.println("Method:  " + methodName);
        System.out.println();

        // File paths
        String originalFile = ORIGINAL_DIR + datasetPrefix + ".arff";
        String syntheticFile = SYNTHETIC_DIR + methodName + "/" + datasetPrefix + "_synthetic.arff";

        // Check if files exist
        if (!new File(originalFile).exists()) {
            System.out.println("✗ Original dataset not found: " + originalFile);
            return;
        }

        if (!new File(syntheticFile).exists()) {
            System.out.println("✗ Synthetic dataset not found: " + syntheticFile);
            System.out.println("  Please generate synthetic data first (Option [2] in main menu)");
            return;
        }

        // Load datasets
        System.out.println("Loading datasets...");
        Instances originalData = loadDataset(originalFile);
        Instances syntheticData = loadDataset(syntheticFile);

        originalData.setClassIndex(originalData.numAttributes() - 1);
        syntheticData.setClassIndex(syntheticData.numAttributes() - 1);

        System.out.println("  Original:  " + originalData.numInstances() + " instances, " +
                         originalData.numAttributes() + " attributes");
        System.out.println("  Synthetic: " + syntheticData.numInstances() + " instances, " +
                         syntheticData.numAttributes() + " attributes");
        System.out.println();

        // Verify structure match
        if (originalData.numAttributes() != syntheticData.numAttributes()) {
            System.out.println("✗ Attribute mismatch between original and synthetic data");
            return;
        }

        // === 1. RE-IDENTIFICATION ATTACK ===
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("  1. Re-Identification Attack");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Create re-identification attack (uses parallel processing, no BallTree needed)
        ReIdentificationAttack reIdAttack = new ReIdentificationAttack(originalData, syntheticData);
        reIdAttack.performAttack();
        reIdAttack.printReport();

        // === 2. LINKAGE ATTACK ===
        System.out.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("  2. Linkage Attack");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        LinkageAttack linkageAttack = new LinkageAttack(originalData, syntheticData);
        linkageAttack.performAttack();
        linkageAttack.printReport();

        // === 3. SAVE RESULTS ===
        saveResults(datasetName, datasetPrefix, methodName, reIdAttack, linkageAttack);

        System.out.println("✓ Privacy evaluation completed!");
    }

    /**
     * Save privacy attack results to files
     */
    private static void saveResults(String datasetName, String datasetPrefix, String methodName,
                                    ReIdentificationAttack reIdAttack, LinkageAttack linkageAttack) throws Exception {
        // Create output directory
        new File(OUTPUT_DIR).mkdirs();

        // CSV file for this dataset/method combination
        String csvFile = OUTPUT_DIR + datasetPrefix + "_" + methodName + "_privacy_results.csv";
        PrintWriter csvWriter = new PrintWriter(new FileWriter(csvFile));

        csvWriter.println("Dataset,Method,Attack_Type,Metric,Value");

        // Re-identification results
        Map<String, Object> reIdStats = reIdAttack.getDetailedStatistics();
        for (Map.Entry<String, Object> entry : reIdStats.entrySet()) {
            if (entry.getValue() instanceof Double) {
                csvWriter.printf("%s,%s,Re-identification,%s,%.6f\n",
                        datasetName, methodName, entry.getKey().replace(",", ";"), entry.getValue());
            }
        }

        // Linkage results
        Map<String, Object> linkageStats = linkageAttack.getDetailedStatistics();
        for (Map.Entry<String, Object> entry : linkageStats.entrySet()) {
            csvWriter.printf("%s,%s,Linkage,%s,%.6f\n",
                    datasetName, methodName, entry.getKey().replace(",", ";"), entry.getValue());
        }

        // Combined risk scores
        // PRS_ReID = 0.7 * ReID@1 + 0.3 * ReID@5
        double reIdAt1 = reIdAttack.getReIDAt1();
        double reIdAt5 = reIdAttack.getReIDAt5();
        double reIdRisk = 0.7 * reIdAt1 + 0.3 * reIdAt5;

        double linkageRisk = linkageAttack.calculatePrivacyRiskScore();
        double combinedRisk = (reIdRisk + linkageRisk) / 2.0;

        csvWriter.printf("%s,%s,Combined,Re-identification_Risk,%.6f\n", datasetName, methodName, reIdRisk);
        csvWriter.printf("%s,%s,Combined,Linkage_Risk,%.6f\n", datasetName, methodName, linkageRisk);
        csvWriter.printf("%s,%s,Combined,Overall_Privacy_Risk,%.6f\n", datasetName, methodName, combinedRisk);

        csvWriter.close();

        System.out.println("\n  ✓ Results saved to: " + csvFile);
    }

    /**
     * Load dataset from file
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    /**
     * Get integer input from user
     */
    private static int getIntInput(Scanner scanner) {
        try {
            return scanner.nextInt();
        } catch (Exception e) {
            scanner.nextLine(); // Clear buffer
            return -1;
        }
    }

    /**
     * Main method - Entry point for privacy attack evaluation
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║     Privacy Attack Evaluator for Synthetic Data              ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");

        showPrivacyAttackMenu(scanner);

        scanner.close();
        System.out.println("\nGoodbye!");
    }
}
