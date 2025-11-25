package privacyguard;

import weka.core.Instance;
import weka.core.Instances;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * Re-identification Attack: Attempts to match synthetic records back to original records
 * using distance-based similarity matching.
 *
 * This simulates an adversary who has access to both original D = {x_1,...,x_n} and
 * synthetic D' = {s_1,...,s_m} datasets and tries to determine which original record
 * corresponds to each synthetic record.
 *
 * FORMAL DEFINITION:
 * - For each synthetic record s_j, compute distance vector m_j = (d(s_j,x_1),...,d(s_j,x_n))
 * - Sort distances in ascending order to get rank r_j of correct source record x_π(j)
 * - Assumption: π(j) = j (index-aligned: s_j generated from x_j)
 * - ReID@K = (number of records with r_j ≤ K) / m
 *
 * OPTIMIZATION: Uses parallel processing across all CPU cores for distance calculations.
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class ReIdentificationAttack {

    private Instances originalData;
    private Instances syntheticData;
    private int classIndex;
    private int numThreads;

    // Results (r_j values for all synthetic records)
    private int[] ranks;  // ranks[j] = r_j (rank of correct source record for s_j)
    private int[] closestMatchIndices;  // Closest match found by attack
    private double[] minDistances;      // Distance to closest match

    /**
     * Constructor with automatic thread detection
     */
    public ReIdentificationAttack(Instances originalData, Instances syntheticData) {
        this(originalData, syntheticData, Runtime.getRuntime().availableProcessors());
    }

    /**
     * Constructor with custom thread count
     * @param originalData Original dataset D = {x_1,...,x_n}
     * @param syntheticData Synthetic dataset D' = {s_1,...,s_m}
     * @param numThreads Number of parallel threads (0 = auto-detect)
     */
    public ReIdentificationAttack(Instances originalData, Instances syntheticData, int numThreads) {
        this.originalData = new Instances(originalData);
        this.syntheticData = new Instances(syntheticData);
        this.classIndex = originalData.classIndex();
        this.numThreads = (numThreads <= 0) ? Runtime.getRuntime().availableProcessors() : numThreads;
    }

    /**
     * Performs the re-identification attack using parallel processing.
     * For each synthetic record s_j:
     *   1. Compute distances to all original records
     *   2. Find rank r_j of correct source record x_j
     */
    public void performAttack() {
        int m = syntheticData.numInstances();  // Number of synthetic records
        int n = originalData.numInstances();   // Number of original records

        ranks = new int[m];
        closestMatchIndices = new int[m];
        minDistances = new double[m];

        System.out.println("\n=== Re-Identification Attack ===");
        System.out.println("Original records (n): " + n);
        System.out.println("Synthetic records (m): " + m);
        System.out.println("Parallel threads: " + numThreads);
        System.out.println("Processing...");

        long startTime = System.currentTimeMillis();

        // Parallel processing using Java Streams
        IntStream.range(0, m)
            .parallel()
            .forEach(synIdx -> {
                Instance syntheticInstance = syntheticData.instance(synIdx);

                // Compute distance vector m_j = (d(s_j, x_1), ..., d(s_j, x_n))
                double[] distances = new double[n];
                for (int origIdx = 0; origIdx < n; origIdx++) {
                    distances[origIdx] = calculateDistance(syntheticInstance,
                                                          originalData.instance(origIdx));
                }

                // Find closest match
                int closestIdx = findMinIndex(distances);
                closestMatchIndices[synIdx] = closestIdx;
                minDistances[synIdx] = distances[closestIdx];

                // Calculate rank r_j of correct source record x_π(j)
                // Assumption: π(j) = j (s_j generated from x_j)
                int correctIdx = synIdx;
                if (correctIdx < n) {
                    ranks[synIdx] = calculateRank(distances, correctIdx);
                } else {
                    ranks[synIdx] = n;  // Out of bounds, assign worst rank
                }

                // Progress reporting (thread-safe)
                if ((synIdx + 1) % 1000 == 0) {
                    synchronized(System.out) {
                        System.out.println("  Processed " + (synIdx + 1) + "/" + m + " records");
                    }
                }
            });

        long endTime = System.currentTimeMillis();
        System.out.println("Attack completed in " + (endTime - startTime) + " ms");
    }

    /**
     * Calculate Euclidean distance between two instances (excluding class attribute)
     * Formula: d(s_j, x_i) = sqrt((1/d') * Σ(s_j^(k) - x_i^(k))^2)
     */
    private double calculateDistance(Instance inst1, Instance inst2) {
        double sumSquaredDiff = 0.0;
        int numAttributes = 0;

        for (int i = 0; i < inst1.numAttributes(); i++) {
            if (i == classIndex) continue;  // Skip class attribute

            double val1 = inst1.value(i);
            double val2 = inst2.value(i);

            if (!Double.isNaN(val1) && !Double.isNaN(val2)) {
                sumSquaredDiff += Math.pow(val1 - val2, 2);
                numAttributes++;
            }
        }

        return numAttributes > 0 ? Math.sqrt(sumSquaredDiff / numAttributes) : Double.MAX_VALUE;
    }

    /**
     * Calculate rank r_j = Rank(m_j, x_correctIdx)
     * Returns the position of x_correctIdx in the sorted distance list
     * Rank 1 = closest match, Rank 2 = second closest, etc.
     */
    private int calculateRank(double[] distances, int correctIdx) {
        if (correctIdx >= distances.length) {
            return distances.length;
        }

        double correctDistance = distances[correctIdx];
        int rank = 1;

        // Count how many records are closer than the correct one
        for (int i = 0; i < distances.length; i++) {
            if (i != correctIdx && distances[i] < correctDistance) {
                rank++;
            }
        }

        return rank;
    }

    /**
     * Find index of minimum value in array
     */
    private int findMinIndex(double[] array) {
        int minIndex = 0;
        double minValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] < minValue) {
                minValue = array[i];
                minIndex = i;
            }
        }

        return minIndex;
    }

    /**
     * Calculate ReID@K for multiple K values in one pass
     * ReID@K = (number of records with r_j ≤ K) / m
     *
     * @return Map of K -> ReID@K values for K ∈ {1, 5, 10, 25, 50, 100}
     */
    public Map<Integer, Double> calculateReIDAtK() {
        if (ranks == null) {
            throw new IllegalStateException("Must perform attack first");
        }

        int[] K_VALUES = {1, 5, 10, 25, 50, 100};
        Map<Integer, Double> results = new LinkedHashMap<>();
        int m = ranks.length;

        for (int K : K_VALUES) {
            int count = 0;
            for (int rank : ranks) {
                if (rank <= K && rank > 0) {
                    count++;
                }
            }
            results.put(K, (double) count / m);
        }

        return results;
    }

    /**
     * Calculate average rank: ACR = (1/m) * Σ r_j
     * Lower ACR = worse privacy (correct matches are close)
     */
    public double calculateAverageRank() {
        if (ranks == null) {
            throw new IllegalStateException("Must perform attack first");
        }

        long sum = 0;
        int count = 0;

        for (int rank : ranks) {
            if (rank > 0) {
                sum += rank;
                count++;
            }
        }

        return count > 0 ? (double) sum / count : 0.0;
    }

    /**
     * Get detailed statistics about the attack results
     */
    public Map<String, Object> getDetailedStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();

        // ReID@K values
        Map<Integer, Double> reidAtK = calculateReIDAtK();
        for (Map.Entry<Integer, Double> entry : reidAtK.entrySet()) {
            stats.put("ReID@" + entry.getKey(), entry.getValue());
        }

        // Average rank
        stats.put("Average Rank (ACR)", calculateAverageRank());

        // Distance statistics
        double avgDistance = 0.0;
        double minDistance = Double.MAX_VALUE;
        double maxDistance = Double.MIN_VALUE;

        for (double dist : minDistances) {
            avgDistance += dist;
            minDistance = Math.min(minDistance, dist);
            maxDistance = Math.max(maxDistance, dist);
        }
        avgDistance /= minDistances.length;

        stats.put("Avg Distance to Closest Match", avgDistance);
        stats.put("Min Distance to Closest Match", minDistance);
        stats.put("Max Distance to Closest Match", maxDistance);

        return stats;
    }

    /**
     * Print a summary report of the attack results
     */
    public void printReport() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("RE-IDENTIFICATION ATTACK REPORT");
        System.out.println("=".repeat(60));
        System.out.println("Dataset: " + originalData.relationName());
        System.out.println("Synthetic records (m): " + syntheticData.numInstances());
        System.out.println("Original records (n): " + originalData.numInstances());
        System.out.println("Features (d'): " + (syntheticData.numAttributes() - (classIndex >= 0 ? 1 : 0)));
        System.out.println();

        Map<String, Object> stats = getDetailedStatistics();

        System.out.println("METRICS:");
        System.out.println("-".repeat(60));
        for (Map.Entry<String, Object> entry : stats.entrySet()) {
            if (entry.getValue() instanceof Double) {
                System.out.printf("%-35s: %.6f\n", entry.getKey(), entry.getValue());
            } else {
                System.out.printf("%-35s: %s\n", entry.getKey(), entry.getValue());
            }
        }

        System.out.println("\n" + "=".repeat(60));
    }

    /**
     * Get the rank array (for external privacy risk score calculation)
     */
    public int[] getRanks() {
        return ranks;
    }

    /**
     * Get ReID@1 (percentage where closest match is correct)
     */
    public double getReIDAt1() {
        return calculateReIDAtK().get(1);
    }

    /**
     * Get ReID@5 (percentage where correct match is in top 5)
     */
    public double getReIDAt5() {
        return calculateReIDAtK().get(5);
    }
}
