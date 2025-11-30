package privacyguard;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

import java.io.*;
import java.util.*;

/**
 * Empirical Complexity Analysis Benchmark
 *
 * This class measures the execution time of different synthetic data generation
 * methods at various data sizes to empirically determine their computational complexity.
 *
 * Results are saved to CSV for plotting and analysis.
 * NOTE: Only timing results are saved - synthetic data is NOT saved to disk.
 *
 * @author Abdulmohsen Almalawi
 */
public class ComplexityBenchmark {

    // Percentages to test (10% to 100%)
    private static final int[] PERCENTAGES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // Number of runs for averaging (to reduce noise)
    private static final int NUM_RUNS = 1;  // Set to 3 for more accurate results

    // All available method names
    private static final String[] ALL_METHODS = {
        "PrivacyGuard",
        "Equal_Width_Binning",
        "kmeans",
        "k_anonymity",
        "LaplaceDP"
    };

    /**
     * Main method for standalone execution
     */
    public static void main(String[] args) {
        String datasetPath = "datasets/Original/Edge-IIoTset.arff";
        String datasetName = "Edge-IIoTset";

        if (args.length >= 2) {
            datasetPath = args[0];
            datasetName = args[1];
        }

        try {
            // Run all methods by default
            runBenchmark(datasetPath, datasetName, ALL_METHODS);
        } catch (Exception e) {
            System.err.println("Error during benchmark: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Run benchmark for ALL methods (called from menu - backward compatible)
     */
    public static void runBenchmark(String datasetPath, String datasetName) throws Exception {
        runBenchmark(datasetPath, datasetName, ALL_METHODS);
    }

    /**
     * Run benchmark for SPECIFIC methods
     *
     * @param datasetPath Path to the ARFF dataset file
     * @param datasetName Name of the dataset (for output file naming)
     * @param methodsToRun Array of method names to benchmark
     */
    public static void runBenchmark(String datasetPath, String datasetName, String[] methodsToRun) throws Exception {
        String outputCsv = "output/complexity_benchmark_" + datasetName + ".csv";

        System.out.println("==============================================");
        System.out.println("  Empirical Complexity Analysis Benchmark");
        System.out.println("==============================================\n");

        // Create output directory if it doesn't exist
        new File("output").mkdirs();

        // Load the full dataset
        System.out.println("Loading dataset: " + datasetPath);
        Instances fullData = loadDataset(datasetPath);
        int totalInstances = fullData.numInstances();
        int numAttributes = fullData.numAttributes();
        System.out.println("Total instances: " + totalInstances);
        System.out.println("Number of attributes: " + numAttributes);
        System.out.println();

        // Show which methods will be tested
        System.out.println("Methods to benchmark: " + String.join(", ", methodsToRun));
        System.out.println();

        // Results storage: method -> percentage -> list of times
        Map<String, Map<Integer, List<Long>>> results = new LinkedHashMap<>();
        for (String method : methodsToRun) {
            results.put(method, new LinkedHashMap<>());
            for (int pct : PERCENTAGES) {
                results.get(method).put(pct, new ArrayList<>());
            }
        }

        // Run benchmarks
        for (int run = 1; run <= NUM_RUNS; run++) {
            System.out.println("========== RUN " + run + " of " + NUM_RUNS + " ==========\n");

            for (int percentage : PERCENTAGES) {
                int sampleSize = (int) Math.round(totalInstances * percentage / 100.0);
                System.out.println("--- Testing with " + percentage + "% data (" + sampleSize + " instances) ---");

                // Get sample of data
                Instances sampleData = getSample(fullData, percentage);

                // Test each selected method
                for (String methodName : methodsToRun) {
                    System.out.print("  " + methodName + ": ");
                    System.out.flush();

                    long startTime = System.currentTimeMillis();
                    runMethod(methodName, sampleData, percentage);
                    long endTime = System.currentTimeMillis();
                    long elapsed = endTime - startTime;

                    results.get(methodName).get(percentage).add(elapsed);
                    System.out.println(elapsed + " ms");
                }
                System.out.println();
            }
        }

        // Save results to CSV (only timing data, NOT synthetic data)
        saveResults(results, totalInstances, outputCsv, methodsToRun);

        // Copy CSV to complexity_plots directory for Python script
        copyToPlotsDirectory(outputCsv);

        System.out.println("\n==============================================");
        System.out.println("  Benchmark Complete!");
        System.out.println("  Results saved to: " + outputCsv);
        System.out.println("==============================================");
    }

    /**
     * Copy CSV file to complexity_plots directory for Python plotting
     */
    private static void copyToPlotsDirectory(String csvPath) {
        try {
            File sourceFile = new File(csvPath);
            // Try both possible locations
            String[] destDirs = {
                "output/complexity_plots",
                "/home/abdul/Desktop/claudeProjects/PrivacyGuard/output/complexity_plots"
            };

            for (String destDir : destDirs) {
                File destDirectory = new File(destDir);
                if (destDirectory.exists() || destDirectory.mkdirs()) {
                    File destFile = new File(destDirectory, sourceFile.getName());

                    // Copy file
                    try (InputStream in = new FileInputStream(sourceFile);
                         OutputStream out = new FileOutputStream(destFile)) {
                        byte[] buffer = new byte[1024];
                        int length;
                        while ((length = in.read(buffer)) > 0) {
                            out.write(buffer, 0, length);
                        }
                    }
                    System.out.println("CSV copied to: " + destFile.getPath());
                }
            }
        } catch (Exception e) {
            System.out.println("Note: Could not copy CSV to plots directory: " + e.getMessage());
        }
    }

    /**
     * Run benchmark for a SINGLE method
     */
    public static void runSingleMethodBenchmark(String datasetPath, String datasetName, String methodName) throws Exception {
        runBenchmark(datasetPath, datasetName, new String[]{methodName});
    }

    /**
     * Get available method names
     */
    public static String[] getAvailableMethods() {
        return ALL_METHODS.clone();
    }

    /**
     * Load dataset from ARFF file
     */
    private static Instances loadDataset(String path) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(path));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Get a random sample of the dataset at the specified percentage
     */
    private static Instances getSample(Instances data, int percentage) throws Exception {
        if (percentage == 100) {
            return new Instances(data);
        }

        Resample resample = new Resample();
        resample.setNoReplacement(true);
        resample.setSampleSizePercent(percentage);
        resample.setRandomSeed((int) System.currentTimeMillis());
        resample.setInputFormat(data);
        return Filter.useFilter(data, resample);
    }

    /**
     * Run a specific synthetic data generation method
     * NOTE: Data is generated in memory only, NOT saved to disk
     *
     * @param methodName Name of the method to run
     * @param data The dataset to process
     * @param percentage The percentage of data (used to scale k-means clusters)
     */
    private static void runMethod(String methodName, Instances data, int percentage) throws Exception {
        Instances dataCopy = new Instances(data);

        switch (methodName) {
            case "PrivacyGuard":
                // Use maxClusterSize=5 to match evaluation settings
                PrivacyGuardGenerator privacyGuard = new PrivacyGuardGenerator(5, 0.5);
                privacyGuard.generateSyntheticData(dataCopy);
                break;

            case "Equal_Width_Binning":
                EqualWidthBinning binning = new EqualWidthBinning(10);
                binning.generateSyntheticData(dataCopy);
                break;

            case "kmeans":
                // Scale clusters based on data percentage: 10%->100, 20%->200, ..., 100%->1000
                int numClusters = percentage * 10;
                KMeansBaseline kmeans = new KMeansBaseline(numClusters);
                kmeans.generateSyntheticData(dataCopy);
                break;

            case "k_anonymity":
                KAnonymity kanonymity = new KAnonymity(5);
                kanonymity.generateSyntheticData(dataCopy);
                break;

            case "LaplaceDP":
                LaplaceDPBaseline laplaceDP = new LaplaceDPBaseline(1.0);
                laplaceDP.generateSyntheticData(dataCopy);
                break;

            default:
                throw new IllegalArgumentException("Unknown method: " + methodName);
        }
        // NOTE: synthetic data is discarded here - only time measurement matters
    }

    /**
     * Save results to CSV file (only timing data)
     * If file exists, reads existing data and updates/appends new method results
     */
    private static void saveResults(Map<String, Map<Integer, List<Long>>> results,
                                    int totalInstances, String outputCsv, String[] methodsToRun) throws IOException {

        // Read existing data if file exists
        Map<String, Map<Integer, Long>> existingData = new LinkedHashMap<>();
        File csvFile = new File(outputCsv);

        if (csvFile.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
                String line = reader.readLine(); // Skip header
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length >= 5) {
                        String method = parts[0];
                        int percentage = Integer.parseInt(parts[1]);
                        long timeMs = Long.parseLong(parts[4]);

                        // Only keep if not in current methods to run (will be replaced)
                        boolean willBeReplaced = false;
                        for (String m : methodsToRun) {
                            if (m.equals(method)) {
                                willBeReplaced = true;
                                break;
                            }
                        }

                        if (!willBeReplaced) {
                            existingData.computeIfAbsent(method, k -> new LinkedHashMap<>());
                            existingData.get(method).put(percentage, timeMs);
                        }
                    }
                }
            } catch (Exception e) {
                System.out.println("Note: Could not read existing CSV, creating new file.");
            }
        }

        // Write all data (existing + new)
        PrintWriter writer = new PrintWriter(new FileWriter(outputCsv));

        // Write header
        writer.println("method,percentage,num_instances,run,time_ms,avg_time_ms");

        // Write existing data first (methods not in current run)
        for (String method : existingData.keySet()) {
            for (int percentage : PERCENTAGES) {
                Long timeMs = existingData.get(method).get(percentage);
                if (timeMs != null) {
                    int numInstances = (int) Math.round(totalInstances * percentage / 100.0);
                    writer.printf("%s,%d,%d,%d,%d,%.2f%n",
                        method,
                        percentage,
                        numInstances,
                        1,
                        timeMs,
                        (double) timeMs);
                }
            }
        }

        // Write new data
        for (String method : methodsToRun) {
            for (int percentage : PERCENTAGES) {
                List<Long> times = results.get(method).get(percentage);
                int numInstances = (int) Math.round(totalInstances * percentage / 100.0);
                double avgTime = times.stream().mapToLong(Long::longValue).average().orElse(0);

                for (int run = 0; run < times.size(); run++) {
                    writer.printf("%s,%d,%d,%d,%d,%.2f%n",
                        method,
                        percentage,
                        numInstances,
                        run + 1,
                        times.get(run),
                        avgTime);
                }
            }
        }

        writer.close();

        // Show what methods are now in the file
        Set<String> allMethods = new LinkedHashSet<>(existingData.keySet());
        for (String m : methodsToRun) {
            allMethods.add(m);
        }
        System.out.println("\nCSV now contains methods: " + String.join(", ", allMethods));

        // Print summary table with DATA SIZE (n) prominently
        System.out.println("\n" + "=".repeat(100));
        System.out.println("  DATA SIZE (n) vs EXECUTION TIME (ms)");
        System.out.println("=".repeat(100));

        // Calculate actual sizes
        int[] sizes = new int[PERCENTAGES.length];
        for (int i = 0; i < PERCENTAGES.length; i++) {
            sizes[i] = (int) Math.round(totalInstances * PERCENTAGES[i] / 100.0);
        }

        // Header with actual sizes
        System.out.printf("%-20s", "Method");
        for (int size : sizes) {
            System.out.printf("%10d", size);
        }
        System.out.println();

        System.out.printf("%-20s", "");
        for (int pct : PERCENTAGES) {
            System.out.printf("%9d%%", pct);
        }
        System.out.println();

        System.out.println("-".repeat(20 + sizes.length * 10));

        // Data rows
        for (String method : methodsToRun) {
            System.out.printf("%-20s", method);
            for (int percentage : PERCENTAGES) {
                List<Long> times = results.get(method).get(percentage);
                double avgTime = times.stream().mapToLong(Long::longValue).average().orElse(0);
                System.out.printf("%10.0f", avgTime);
            }
            System.out.println();
        }

        System.out.println("\n(Times in milliseconds)");
    }
}
