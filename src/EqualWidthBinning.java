package privacyguard;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.*;

/**
 * Equal-Width Binning Baseline
 *
 * Simple discretization method that divides each feature into equal-width bins.
 * This serves as a naive baseline to demonstrate that simple anonymization
 * approaches provide insufficient privacy-utility balance.
 *
 * Method:
 * 1. For each feature, find min and max values
 * 2. Divide range into N equal-width bins
 * 3. Replace each value with its bin ID
 *
 * Parameters:
 * - numBins: Number of bins per feature (default: 10)
 *
 * Expected Performance:
 * - Privacy: Poor (large bins, easy to reverse)
 * - Utility: Low-Medium (loses precision)
 * - Speed: Fast (simple computation)
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class EqualWidthBinning {

    private int numBins;
    private Map<Integer, BinInfo> featureBins;  // featureIndex -> BinInfo

    /**
     * Bin information for a feature
     */
    private static class BinInfo {
        double min;
        double max;
        double binWidth;
        int numBins;

        BinInfo(double min, double max, int numBins) {
            this.min = min;
            this.max = max;
            this.numBins = numBins;
            // Add small epsilon to avoid edge cases
            this.binWidth = (max - min + 0.0001) / numBins;
        }

        int getBinID(double value) {
            if (value <= min) return 0;
            if (value >= max) return numBins - 1;

            int binID = (int) ((value - min) / binWidth);
            // Ensure within bounds
            return Math.min(binID, numBins - 1);
        }
    }

    /**
     * Constructor with default number of bins
     */
    public EqualWidthBinning() {
        this(10);  // Default: 10 bins
    }

    /**
     * Constructor with custom number of bins
     * @param numBins Number of bins per feature
     */
    public EqualWidthBinning(int numBins) {
        this.numBins = numBins;
        this.featureBins = new HashMap<>();
    }

    /**
     * Generate synthetic data using equal-width binning
     *
     * @param data Original dataset
     * @return Synthetic dataset with binned values
     */
    public Instances generateSyntheticData(Instances data) throws Exception {
        System.out.println("  Equal-Width Binning: Analyzing features...");

        // Determine class index
        int classIndex = data.classIndex();

        // Create copy of data
        Instances syntheticData = new Instances(data);

        // Step 1: Learn bin boundaries for each feature
        learnBins(data, classIndex);

        // Step 2: Replace values with bin IDs
        System.out.println("  Equal-Width Binning: Applying binning...");
        for (int i = 0; i < syntheticData.numInstances(); i++) {
            for (int featureIdx = 0; featureIdx < syntheticData.numAttributes(); featureIdx++) {
                // Skip class attribute
                if (featureIdx == classIndex) continue;

                double originalValue = syntheticData.instance(i).value(featureIdx);
                BinInfo binInfo = featureBins.get(featureIdx);

                if (binInfo != null) {
                    int binID = binInfo.getBinID(originalValue);
                    syntheticData.instance(i).setValue(featureIdx, binID);
                }
            }

            if ((i + 1) % 10000 == 0) {
                System.out.println("    Processed " + (i + 1) + "/" + syntheticData.numInstances() + " records");
            }
        }

        System.out.println("  Equal-Width Binning: Complete!");
        return syntheticData;
    }

    /**
     * Learn bin boundaries for each feature
     */
    private void learnBins(Instances data, int classIndex) {
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

            // Store bin info
            featureBins.put(featureIdx, new BinInfo(min, max, numBins));
        }

        System.out.println("    Learned " + featureBins.size() + " feature bins (numBins=" + numBins + ")");
    }

    /**
     * Get method name
     */
    public String getMethodName() {
        return "Equal-Width Binning (numBins=" + numBins + ")";
    }

    /**
     * Get method description
     */
    public String getDescription() {
        return "Simple discretization: divides each feature into " + numBins + " equal-width bins";
    }

    /**
     * Get parameters as string
     */
    public String getParameters() {
        return "numBins=" + numBins;
    }
}
