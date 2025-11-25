package privacyguard;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.*;

/**
 * k-Anonymity Baseline
 *
 * Implements a simplified k-anonymity approach through micro-aggregation.
 * Groups similar records and replaces them with group centroids to ensure
 * each equivalence class has at least k members.
 *
 * Method:
 * 1. Sort records by distance to dataset centroid
 * 2. Group consecutive records into groups of size k
 * 3. Replace each group with their centroid
 *
 * Parameters:
 * - k: Minimum equivalence class size (default: 5)
 *
 * Expected Performance:
 * - Privacy: Good (guarantees k-anonymity)
 * - Utility: Low-Medium (significant generalization)
 * - Speed: Slow (distance computations)
 *
 * Note:
 * This is a simplified implementation. Full k-anonymity would use
 * more sophisticated generalization hierarchies and suppression.
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class KAnonymity {

    private int k;  // Anonymity parameter

    /**
     * Constructor with default k
     */
    public KAnonymity() {
        this(5);  // Default: k=5
    }

    /**
     * Constructor with custom k
     * @param k Minimum equivalence class size
     */
    public KAnonymity(int k) {
        this.k = k;
    }

    /**
     * Generate synthetic data using k-anonymity micro-aggregation
     *
     * @param data Original dataset
     * @return Synthetic dataset satisfying k-anonymity
     */
    public Instances generateSyntheticData(Instances data) throws Exception {
        System.out.println("  k-anonymity: Starting micro-aggregation (k=" + k + ")...");

        // Determine class index
        int classIndex = data.classIndex();

        // Save class labels
        double[] classLabels = new double[data.numInstances()];
        if (classIndex >= 0) {
            for (int i = 0; i < data.numInstances(); i++) {
                classLabels[i] = data.instance(i).classValue();
            }
        }

        // Create copy without class attribute for processing
        Instances dataNoClass = removeClassAttribute(data, classIndex);

        // Step 1: Calculate dataset centroid
        System.out.println("    Calculating dataset centroid...");
        double[] centroid = calculateCentroid(dataNoClass);

        // Step 2: Calculate distances to centroid
        System.out.println("    Calculating distances to centroid...");
        List<InstanceWithDistance> instancesWithDist = new ArrayList<>();
        for (int i = 0; i < dataNoClass.numInstances(); i++) {
            double dist = euclideanDistance(dataNoClass.instance(i), centroid);
            instancesWithDist.add(new InstanceWithDistance(i, dist));
        }

        // Step 3: Sort by distance
        System.out.println("    Sorting records by distance...");
        Collections.sort(instancesWithDist, Comparator.comparingDouble(iwd -> iwd.distance));

        // Step 4: Create groups and replace with centroids
        System.out.println("    Creating k-anonymous groups...");
        Instances syntheticData = new Instances(data);

        int numGroups = dataNoClass.numInstances() / k;
        int remainder = dataNoClass.numInstances() % k;

        int instanceIdx = 0;
        for (int group = 0; group < numGroups; group++) {
            // Collect k instances for this group
            List<Integer> groupIndices = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                groupIndices.add(instancesWithDist.get(instanceIdx).originalIndex);
                instanceIdx++;
            }

            // Calculate group centroid
            double[] groupCentroid = calculateGroupCentroid(dataNoClass, groupIndices);

            // Replace all instances in group with centroid
            for (int idx : groupIndices) {
                for (int attrIdx = 0; attrIdx < syntheticData.numAttributes(); attrIdx++) {
                    if (attrIdx == classIndex) continue;  // Skip class
                    syntheticData.instance(idx).setValue(attrIdx, groupCentroid[attrIdx]);
                }
            }

            if ((group + 1) % 1000 == 0) {
                System.out.println("      Processed " + (group + 1) + "/" + numGroups + " groups");
            }
        }

        // Handle remainder by adding to last group
        if (remainder > 0) {
            System.out.println("    Handling " + remainder + " remainder records...");
            List<Integer> lastGroupIndices = new ArrayList<>();
            for (int i = 0; i < remainder; i++) {
                lastGroupIndices.add(instancesWithDist.get(instanceIdx).originalIndex);
                instanceIdx++;
            }

            // Add to last complete group's centroid
            double[] lastCentroid = calculateGroupCentroid(dataNoClass, lastGroupIndices);
            for (int idx : lastGroupIndices) {
                for (int attrIdx = 0; attrIdx < syntheticData.numAttributes(); attrIdx++) {
                    if (attrIdx == classIndex) continue;
                    syntheticData.instance(idx).setValue(attrIdx, lastCentroid[attrIdx]);
                }
            }
        }

        // Restore class labels
        if (classIndex >= 0) {
            for (int i = 0; i < syntheticData.numInstances(); i++) {
                syntheticData.instance(i).setClassValue(classLabels[i]);
            }
        }

        System.out.println("  k-anonymity: Complete! Created " + numGroups + " groups");
        return syntheticData;
    }

    /**
     * Remove class attribute from dataset
     */
    private Instances removeClassAttribute(Instances data, int classIndex) {
        if (classIndex < 0) return new Instances(data);

        Instances result = new Instances(data);
        result.setClassIndex(-1);
        return result;
    }

    /**
     * Calculate centroid of entire dataset
     */
    private double[] calculateCentroid(Instances data) {
        double[] centroid = new double[data.numAttributes()];

        for (int attrIdx = 0; attrIdx < data.numAttributes(); attrIdx++) {
            double sum = 0;
            int count = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                double val = data.instance(i).value(attrIdx);
                if (!Double.isNaN(val)) {
                    sum += val;
                    count++;
                }
            }
            centroid[attrIdx] = count > 0 ? sum / count : 0;
        }

        return centroid;
    }

    /**
     * Calculate centroid of a group of instances
     */
    private double[] calculateGroupCentroid(Instances data, List<Integer> indices) {
        double[] centroid = new double[data.numAttributes()];

        for (int attrIdx = 0; attrIdx < data.numAttributes(); attrIdx++) {
            double sum = 0;
            int count = 0;
            for (int idx : indices) {
                double val = data.instance(idx).value(attrIdx);
                if (!Double.isNaN(val)) {
                    sum += val;
                    count++;
                }
            }
            centroid[attrIdx] = count > 0 ? sum / count : 0;
        }

        return centroid;
    }

    /**
     * Calculate Euclidean distance between instance and centroid
     */
    private double euclideanDistance(weka.core.Instance instance, double[] centroid) {
        double sum = 0;
        int count = 0;

        for (int i = 0; i < instance.numAttributes(); i++) {
            double val = instance.value(i);
            if (!Double.isNaN(val)) {
                sum += Math.pow(val - centroid[i], 2);
                count++;
            }
        }

        return count > 0 ? Math.sqrt(sum / count) : 0;
    }

    /**
     * Helper class to store instance index with distance
     */
    private static class InstanceWithDistance {
        int originalIndex;
        double distance;

        InstanceWithDistance(int originalIndex, double distance) {
            this.originalIndex = originalIndex;
            this.distance = distance;
        }
    }

    /**
     * Get method name
     */
    public String getMethodName() {
        return "k-Anonymity (k=" + k + ")";
    }

    /**
     * Get method description
     */
    public String getDescription() {
        return "Micro-aggregation to ensure k-anonymity with k=" + k;
    }

    /**
     * Get parameters as string
     */
    public String getParameters() {
        return "k=" + k;
    }
}
