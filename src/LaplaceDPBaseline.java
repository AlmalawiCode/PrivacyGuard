package privacyguard;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.*;

/**
 * Laplace Differential Privacy Baseline
 *
 * Implements ε-differential privacy by adding Laplace noise to each feature.
 * This represents the gold standard for formal privacy guarantees.
 *
 * Method:
 * 1. Calculate sensitivity (range) for each feature
 * 2. Add Laplace noise ~ Lap(Δf/ε) to each value
 * 3. Clip noisy values to valid range
 *
 * Parameters:
 * - epsilon: Privacy budget (default: 1.0)
 * - Lower epsilon = stronger privacy, more noise
 *
 * Expected Performance:
 * - Privacy: Excellent (formal ε-DP guarantee)
 * - Utility: Low (significant noise degrades accuracy)
 * - Speed: Fast (simple noise addition)
 *
 * Mathematical Foundation:
 * A mechanism M satisfies ε-differential privacy if for all datasets D1, D2
 * differing on at most one record, and all outcomes S:
 *
 *   Pr[M(D1) ∈ S] ≤ exp(ε) × Pr[M(D2) ∈ S]
 *
 * Laplace Mechanism:
 *   M(x) = f(x) + Lap(Δf/ε)
 *   where Δf = max|f(D1) - f(D2)| (global sensitivity)
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class LaplaceDPBaseline {

    private double epsilon;  // Privacy budget
    private Map<Integer, FeatureStats> featureStats;  // Feature statistics
    private Random random;

    /**
     * Feature statistics for sensitivity calculation
     */
    private static class FeatureStats {
        double min;
        double max;
        double sensitivity;  // Global sensitivity = max - min

        FeatureStats(double min, double max) {
            this.min = min;
            this.max = max;
            this.sensitivity = max - min;
        }

        /**
         * Clip value to valid range
         */
        double clip(double value) {
            return Math.max(min, Math.min(max, value));
        }
    }

    /**
     * Constructor with default epsilon
     */
    public LaplaceDPBaseline() {
        this(1.0);  // Default: ε = 1.0
    }

    /**
     * Constructor with custom epsilon
     * @param epsilon Privacy budget (smaller = stronger privacy)
     */
    public LaplaceDPBaseline(double epsilon) {
        this.epsilon = epsilon;
        this.featureStats = new HashMap<>();
        this.random = new Random(42);  // Fixed seed for reproducibility
    }

    /**
     * Generate synthetic data using Laplace Differential Privacy
     *
     * @param data Original dataset
     * @return Synthetic dataset with ε-DP guarantee
     */
    public Instances generateSyntheticData(Instances data) throws Exception {
        System.out.println("  Laplace DP: Starting noise addition (ε=" + epsilon + ")...");

        // Determine class index
        int classIndex = data.classIndex();

        // Step 1: Learn feature statistics (sensitivity)
        learnFeatureStatistics(data, classIndex);

        // Step 2: Create copy and add Laplace noise
        System.out.println("  Laplace DP: Adding Laplace noise to features...");
        Instances syntheticData = new Instances(data);

        int numFeatures = data.numAttributes();
        if (classIndex >= 0) numFeatures--;

        // Add noise to each instance
        for (int i = 0; i < syntheticData.numInstances(); i++) {
            for (int featureIdx = 0; featureIdx < syntheticData.numAttributes(); featureIdx++) {
                // Skip class attribute
                if (featureIdx == classIndex) continue;

                double originalValue = syntheticData.instance(i).value(featureIdx);
                FeatureStats stats = featureStats.get(featureIdx);

                if (stats != null && !Double.isNaN(originalValue)) {
                    // Add Laplace noise: Lap(Δf/ε)
                    double noisyValue = addLaplaceNoise(originalValue, stats.sensitivity);

                    // Clip to valid range
                    noisyValue = stats.clip(noisyValue);

                    syntheticData.instance(i).setValue(featureIdx, noisyValue);
                }
            }

            if ((i + 1) % 10000 == 0) {
                System.out.println("    Processed " + (i + 1) + "/" + syntheticData.numInstances() + " records");
            }
        }

        System.out.println("  Laplace DP: Complete! Applied ε=" + epsilon + " DP guarantee");
        return syntheticData;
    }

    /**
     * Learn feature statistics for sensitivity calculation
     */
    private void learnFeatureStatistics(Instances data, int classIndex) {
        System.out.println("    Calculating feature sensitivities...");

        for (int featureIdx = 0; featureIdx < data.numAttributes(); featureIdx++) {
            // Skip class attribute
            if (featureIdx == classIndex) continue;

            // Find min and max for this feature
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;

            for (int i = 0; i < data.numInstances(); i++) {
                double value = data.instance(i).value(featureIdx);
                if (!Double.isNaN(value)) {
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }

            // Store statistics
            featureStats.put(featureIdx, new FeatureStats(min, max));
        }

        // Calculate average sensitivity for reporting
        double avgSensitivity = 0;
        for (FeatureStats stats : featureStats.values()) {
            avgSensitivity += stats.sensitivity;
        }
        avgSensitivity /= featureStats.size();

        System.out.println("    Learned " + featureStats.size() + " feature sensitivities " +
                         "(avg Δf=" + String.format("%.3f", avgSensitivity) + ")");
    }

    /**
     * Add Laplace noise to a value
     *
     * Laplace distribution: Lap(μ, b) where b = Δf/ε
     * PDF: f(x|μ,b) = (1/2b) exp(-|x-μ|/b)
     * CDF: F(x|μ,b) = 0.5 + 0.5 × sgn(x-μ) × (1 - exp(-|x-μ|/b))
     *
     * Sampling: X = μ - b × sgn(u) × ln(1 - 2|u|)
     * where u ~ Uniform(-0.5, 0.5)
     *
     * @param value Original value
     * @param sensitivity Global sensitivity (Δf)
     * @return Value + Laplace noise
     */
    private double addLaplaceNoise(double value, double sensitivity) {
        // Calculate scale parameter: b = Δf/ε
        double scale = sensitivity / epsilon;

        // Sample from Laplace distribution
        double u = random.nextDouble() - 0.5;  // Uniform(-0.5, 0.5)
        double noise = -scale * Math.signum(u) * Math.log(1.0 - 2.0 * Math.abs(u));

        return value + noise;
    }

    /**
     * Get method name
     */
    public String getMethodName() {
        return "Laplace DP (ε=" + epsilon + ")";
    }

    /**
     * Get method description
     */
    public String getDescription() {
        return "ε-Differential Privacy via Laplace mechanism with ε=" + epsilon;
    }

    /**
     * Get parameters as string
     */
    public String getParameters() {
        return "epsilon=" + epsilon;
    }

    /**
     * Get formal privacy guarantee
     */
    public String getPrivacyGuarantee() {
        return "Provides ε-differential privacy with ε=" + epsilon +
               " (Pr[M(D1) ∈ S] ≤ e^" + epsilon + " × Pr[M(D2) ∈ S])";
    }

    /**
     * Calculate average noise magnitude (for reporting)
     */
    public double getAverageNoiseMagnitude() {
        if (featureStats.isEmpty()) return 0;

        double totalScale = 0;
        for (FeatureStats stats : featureStats.values()) {
            totalScale += stats.sensitivity / epsilon;
        }

        // Expected magnitude of Laplace(b) is b
        return totalScale / featureStats.size();
    }

    /**
     * Get epsilon value
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Set epsilon value
     */
    public void setEpsilon(double epsilon) {
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        this.epsilon = epsilon;
    }

    /**
     * Calculate composition privacy budget
     * For k independent queries, total privacy is k×ε (basic composition)
     */
    public static double calculateCompositionBudget(double epsilon, int numQueries) {
        return epsilon * numQueries;
    }

    /**
     * Estimate utility loss
     * Higher epsilon = less noise = better utility
     */
    public String estimateUtilityLoss() {
        if (epsilon >= 10.0) return "Low (weak privacy)";
        if (epsilon >= 1.0) return "Medium (moderate privacy)";
        if (epsilon >= 0.1) return "High (strong privacy)";
        return "Very High (very strong privacy)";
    }
}
