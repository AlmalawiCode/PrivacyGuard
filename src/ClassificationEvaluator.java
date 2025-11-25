package privacyguard;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.*;

/**
 * Classification Evaluator (Indices-Based)
 * Trains classifiers on Original and Synthetic data from all methods
 * Uses indices files for consistent train/test splits
 * Saves results in CSV format
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class ClassificationEvaluator {

    private static final String ORIGINAL_DIR = "datasets/Original/";
    private static final String SYNTHETIC_DIR = "output/1_synthetic_data/";
    private static final String INDICES_DIR = "output/2_indices/";
    private static final String OUTPUT_DIR = "output/3_classification/";

    private static final String[] DATASETS = {
        "bot_loT",
        "CICIoT2023",
        "MQTTset",
        "Edge-IIoTset"
    };

    private static final String[] METHODS = {
        "Original",
        "PrivacyGuard",
        "Equal_Width_Binning",
        "KMeans",
        "KAnonymity",
        "LaplacianDP"
    };

    private static final String[] CLASSIFIER_NAMES = {"J48", "RandomForest", "NaiveBayes"};

    /**
     * Run classification evaluation for a specific dataset
     */
    public static void evaluateDataset(String datasetPrefix) throws Exception {
        System.out.println("\n╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║  Classification Evaluation: " + padRight(datasetPrefix, 37) + "║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Create output directory if it doesn't exist
        new File(OUTPUT_DIR).mkdirs();

        // Load indices files
        String trainIndicesFile = INDICES_DIR + datasetPrefix + "_indices_training.txt";
        String testIndicesFile = INDICES_DIR + datasetPrefix + "_indices_testing.txt";

        if (!new File(trainIndicesFile).exists() || !new File(testIndicesFile).exists()) {
            System.err.println("✗ Indices files not found for dataset: " + datasetPrefix);
            System.err.println("  Expected: " + trainIndicesFile);
            System.err.println("  Expected: " + testIndicesFile);
            System.err.println("  Please generate indices first using option [5] in main menu.");
            return;
        }

        System.out.println("Loading indices...");
        List<Integer> trainIndices = loadIndices(trainIndicesFile);
        List<Integer> testIndices = loadIndices(testIndicesFile);
        System.out.println("  Training indices: " + trainIndices.size());
        System.out.println("  Testing indices:  " + testIndices.size());
        System.out.println();

        // Output CSV file
        String outputFile = OUTPUT_DIR + "evaluation_comparison_results_" + datasetPrefix + ".csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            // Write CSV header
            writer.println("Dataset,Classifier,Metric,Class,Value");

            // Evaluate each method
            for (String method : METHODS) {
                System.out.println("═══════════════════════════════════════════════════════════════");
                System.out.println("  Method: " + method);
                System.out.println("═══════════════════════════════════════════════════════════════");

                // Get data file path
                String dataFile;

                if (method.equals("Original")) {
                    dataFile = ORIGINAL_DIR + datasetPrefix + ".arff";
                } else {
                    dataFile = SYNTHETIC_DIR + method + "/" + datasetPrefix + "_synthetic.arff";
                }

                // Check if file exists
                if (!new File(dataFile).exists()) {
                    System.out.println("  ✗ Data file not found: " + dataFile);
                    System.out.println("  ⊳ Skipping method: " + method);
                    System.out.println();
                    continue;
                }

                // Load complete dataset
                System.out.println("  Loading data: " + dataFile);
                Instances completeData = loadDataset(dataFile);
                completeData.setClassIndex(completeData.numAttributes() - 1);

                System.out.println("    Total instances: " + completeData.numInstances());

                // Extract training and testing subsets using indices
                System.out.println("  Extracting train/test splits using indices...");
                Instances trainData = extractInstancesByIndices(completeData, trainIndices);
                Instances testData = extractInstancesByIndices(completeData, testIndices);

                trainData.setClassIndex(trainData.numAttributes() - 1);
                testData.setClassIndex(testData.numAttributes() - 1);

                System.out.println("    Training: " + trainData.numInstances() + " instances");
                System.out.println("    Testing:  " + testData.numInstances() + " instances");
                System.out.println();

                // Evaluate with each classifier
                for (String classifierName : CLASSIFIER_NAMES) {
                    System.out.println("  Classifier: " + classifierName);

                    try {
                        Classifier classifier = getClassifier(classifierName);

                        // Train and evaluate
                        long trainStartTime = System.nanoTime();
                        classifier.buildClassifier(trainData);
                        long trainEndTime = System.nanoTime();
                        long trainingTime = trainEndTime - trainStartTime;

                        // Test
                        long testStartTime = System.nanoTime();
                        Evaluation eval = new Evaluation(trainData);
                        eval.evaluateModel(classifier, testData);
                        long testEndTime = System.nanoTime();
                        long testingTime = testEndTime - testStartTime;

                        // Write timing results
                        writer.printf("%s,%s,Training Time (ns),,%d.0000%n", method, classifierName, trainingTime);
                        writer.printf("%s,%s,Testing Time (ns),,%d.0000%n", method, classifierName, testingTime);

                        System.out.println("    Training Time: " + String.format("%.2f", trainingTime / 1e9) + " seconds");
                        System.out.println("    Testing Time:  " + String.format("%.2f", testingTime / 1e9) + " seconds");

                        // Write per-class metrics
                        for (int i = 0; i < testData.numClasses(); i++) {
                            String className = testData.classAttribute().value(i);

                            // Get metrics (handle case where class might not be detected)
                            double precision = (eval.numTruePositives(i) + eval.numFalsePositives(i)) > 0
                                ? eval.precision(i) : 0.0;
                            double recall = (eval.numTruePositives(i) + eval.numFalseNegatives(i)) > 0
                                ? eval.recall(i) : 0.0;
                            double fMeasure = (precision + recall) > 0
                                ? eval.fMeasure(i) : 0.0;
                            double fpr = (eval.numFalsePositives(i) + eval.numTrueNegatives(i)) > 0
                                ? eval.falsePositiveRate(i) : 0.0;

                            // Replace NaN with 0.0
                            if (Double.isNaN(precision)) precision = 0.0;
                            if (Double.isNaN(recall)) recall = 0.0;
                            if (Double.isNaN(fMeasure)) fMeasure = 0.0;
                            if (Double.isNaN(fpr)) fpr = 0.0;

                            writer.printf("%s,%s,Precision,%s,%.4f%n", method, classifierName, className, precision);
                            writer.printf("%s,%s,Recall,%s,%.4f%n", method, classifierName, className, recall);
                            writer.printf("%s,%s,F-Measure,%s,%.4f%n", method, classifierName, className, fMeasure);
                            writer.printf("%s,%s,False Positive Rate,%s,%.4f%n", method, classifierName, className, fpr);
                        }

                        // Write weighted average metrics
                        double weightedPrecision = eval.weightedPrecision();
                        double weightedRecall = eval.weightedRecall();
                        double weightedFMeasure = eval.weightedFMeasure();
                        double weightedFPR = eval.weightedFalsePositiveRate();

                        // Replace NaN with 0.0
                        if (Double.isNaN(weightedPrecision)) weightedPrecision = 0.0;
                        if (Double.isNaN(weightedRecall)) weightedRecall = 0.0;
                        if (Double.isNaN(weightedFMeasure)) weightedFMeasure = 0.0;
                        if (Double.isNaN(weightedFPR)) weightedFPR = 0.0;

                        writer.printf("%s,%s,Precision,Weighted Avg,%.4f%n", method, classifierName, weightedPrecision);
                        writer.printf("%s,%s,Recall,Weighted Avg,%.4f%n", method, classifierName, weightedRecall);
                        writer.printf("%s,%s,F-Measure,Weighted Avg,%.4f%n", method, classifierName, weightedFMeasure);
                        writer.printf("%s,%s,False Positive Rate,Weighted Avg,%.4f%n", method, classifierName, weightedFPR);

                        System.out.println("    ✓ Evaluation completed");

                    } catch (Exception e) {
                        System.err.println("    ✗ Error evaluating " + classifierName + ": " + e.getMessage());
                    }

                    System.out.println();
                }
            }

            System.out.println("═══════════════════════════════════════════════════════════════");
            System.out.println("✓ Results saved to: " + outputFile);
            System.out.println("═══════════════════════════════════════════════════════════════");
        }
    }

    /**
     * Get classifier instance by name
     */
    private static Classifier getClassifier(String name) throws Exception {
        switch (name) {
            case "J48":
                return new J48();
            case "RandomForest":
                RandomForest rf = new RandomForest();
                rf.setNumIterations(100); // Default 100 trees
                return rf;
            case "NaiveBayes":
                return new NaiveBayes();
            default:
                throw new IllegalArgumentException("Unknown classifier: " + name);
        }
    }

    /**
     * Load dataset from file
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    /**
     * Load indices from file (one index per line, 0-based)
     */
    private static List<Integer> loadIndices(String filePath) throws IOException {
        List<Integer> indices = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    indices.add(Integer.parseInt(line));
                }
            }
        }
        return indices;
    }

    /**
     * Extract instances from dataset based on indices
     */
    private static Instances extractInstancesByIndices(Instances data, List<Integer> indices) {
        // Create new Instances object with the same structure
        Instances subset = new Instances(data, 0);

        // Add instances at specified indices
        for (int index : indices) {
            if (index >= 0 && index < data.numInstances()) {
                subset.add(data.instance(index));
            } else {
                System.err.println("    ⚠ Warning: Index " + index + " out of bounds (0-" + (data.numInstances()-1) + ")");
            }
        }

        return subset;
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
     * Main method for standalone execution
     */
    public static void main(String[] args) throws Exception {
        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║           Classification Evaluation - Standalone             ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");

        if (args.length > 0) {
            // Evaluate specific dataset
            evaluateDataset(args[0]);
        } else {
            // Evaluate all datasets
            for (String dataset : DATASETS) {
                evaluateDataset(dataset);
                System.out.println();
            }
        }
    }
}
