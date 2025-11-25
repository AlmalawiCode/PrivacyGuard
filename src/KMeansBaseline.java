package privacyguard;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.*;

/**
 * k-means per Feature Baseline
 *
 * Applies standard k-means clustering to each feature independently.
 * This serves as an ablation study to demonstrate why VWC (Variable-Width Clustering)
 * is superior to standard k-means.
 *
 * Method:
 * 1. For each feature, apply k-means clustering
 * 2. Replace each value with its cluster ID
 * 3. Fixed number of clusters (k) for all features
 *
 * Parameters:
 * - k: Number of clusters per feature (default: 1000)
 *
 * Expected Performance:
 * - Privacy: Good (similar to VWC)
 * - Utility: Medium-High (slightly worse than VWC)
 * - Speed: Medium (k-means iterations)
 *
 * Comparison with VWC:
 * - VWC adapts cluster count to data distribution
 * - k-means uses fixed k for all features
 * - VWC considers proximity during assignment
 * - k-means uses global optimization
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class KMeansBaseline {

    private int k;  // Number of clusters

    /**
     * Constructor with default k
     */
    public KMeansBaseline() {
        this(1000);  // Default: 1000 clusters (more appropriate for large datasets)
    }

    /**
     * Constructor with custom k
     * @param k Number of clusters per feature
     */
    public KMeansBaseline(int k) {
        this.k = k;
    }

    /**
     * Generate synthetic data using k-means clustering per feature
     *
     * @param data Original dataset
     * @return Synthetic dataset with cluster IDs
     */
    public Instances generateSyntheticData(Instances data) throws Exception {
        System.out.println("  k-means: Starting feature-wise clustering...");

        // Determine class index
        int classIndex = data.classIndex();

        // Create copy of data
        Instances syntheticData = new Instances(data);

        // Apply k-means to each feature
        int numFeatures = data.numAttributes();
        if (classIndex >= 0) numFeatures--;  // Exclude class attribute

        int processedFeatures = 0;
        for (int featureIdx = 0; featureIdx < data.numAttributes(); featureIdx++) {
            // Skip class attribute
            if (featureIdx == classIndex) continue;

            processedFeatures++;
            System.out.println("    Processing feature " + processedFeatures + "/" + numFeatures +
                             " (" + data.attribute(featureIdx).name() + ")");

            // Cluster this feature
            int[] clusterAssignments = clusterFeature(data, featureIdx);

            // Replace values with cluster IDs
            for (int i = 0; i < syntheticData.numInstances(); i++) {
                syntheticData.instance(i).setValue(featureIdx, clusterAssignments[i]);
            }
        }

        System.out.println("  k-means: Complete!");
        return syntheticData;
    }

    /**
     * Apply k-means clustering to a single feature
     */
    private int[] clusterFeature(Instances data, int featureIdx) throws Exception {
        // Create single-feature dataset
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add((Attribute) data.attribute(featureIdx).copy());

        Instances singleFeatureData = new Instances("SingleFeature", attributes, data.numInstances());
        singleFeatureData.setClassIndex(-1);

        // Populate with feature values
        for (int i = 0; i < data.numInstances(); i++) {
            double[] values = new double[1];
            values[0] = data.instance(i).value(featureIdx);
            singleFeatureData.add(new DenseInstance(1.0, values));
        }

        // Apply k-means
        SimpleKMeans kmeans = new SimpleKMeans();

        // Determine actual k (min of desired k and unique values)
        int uniqueValues = countUniqueValues(singleFeatureData, 0);
        int actualK = Math.min(k, uniqueValues);

        kmeans.setNumClusters(actualK);
        kmeans.setMaxIterations(100);
        kmeans.setNumExecutionSlots(1);  // IMPORTANT: Single-threaded for accurate timing comparison
        kmeans.setDistanceFunction(new weka.core.EuclideanDistance(singleFeatureData));
        kmeans.buildClusterer(singleFeatureData);

        // Get cluster assignments
        int[] assignments = new int[data.numInstances()];
        for (int i = 0; i < singleFeatureData.numInstances(); i++) {
            assignments[i] = kmeans.clusterInstance(singleFeatureData.instance(i));
        }

        return assignments;
    }

    /**
     * Count unique values in a feature
     */
    private int countUniqueValues(Instances data, int attrIndex) {
        Set<Double> uniqueValues = new HashSet<>();
        for (int i = 0; i < data.numInstances(); i++) {
            uniqueValues.add(data.instance(i).value(attrIndex));
        }
        return uniqueValues.size();
    }

    /**
     * Get method name
     */
    public String getMethodName() {
        return "k-means per Feature (k=" + k + ")";
    }

    /**
     * Get method description
     */
    public String getDescription() {
        return "Standard k-means clustering applied independently to each feature with k=" + k;
    }

    /**
     * Get parameters as string
     */
    public String getParameters() {
        return "k=" + k;
    }
}
