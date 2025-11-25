package privacyguard;

import java.io.File;
import java.util.ArrayList;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import wekaknnvwc.ProfTools;
import wekaknnvwc.VWC;

/**
 * PrivacyGuard (VWC) Synthetic Data Generator
 *
 * Uses Variable-Width Clustering (VWC) to generate privacy-preserving synthetic data.
 * Each feature is clustered independently, and original values are replaced with cluster IDs.
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class PrivacyGuardGenerator {

    private int maxClusterSize;
    private double betaPercentage;

    /**
     * Constructor with default parameters
     */
    public PrivacyGuardGenerator() {
        this.maxClusterSize = 50;      // Default: max 5 instances per cluster
        this.betaPercentage = 0.5;    // Default: 0.1% of data for beta calculation
    }

    /**
     * Constructor with custom parameters
     */
    public PrivacyGuardGenerator(int maxClusterSize, double betaPercentage) {
        this.maxClusterSize = maxClusterSize;
        this.betaPercentage = betaPercentage;
    }

    /**
     * Generate synthetic data using PrivacyGuard (VWC)
     *
     * @param data Input dataset
     * @return Synthetic dataset with cluster IDs
     */
    public Instances generateSyntheticData(Instances data) throws Exception {
        // Make a copy to avoid modifying original
        Instances syntheticData = new Instances(data);

        // Determine if there is a class attribute
        int classIndex = syntheticData.classIndex();
        boolean hasClassAttribute = classIndex != -1;

        // Save the class attribute if it exists
        Instances originalDataWithClass = null;
        if (hasClassAttribute) {
            originalDataWithClass = new Instances(syntheticData);
        }

        // Iterate over each attribute to perform clustering
        for (int featureIndex = 0; featureIndex < syntheticData.numAttributes(); featureIndex++) {
            // Skip the class attribute if it exists
            if (featureIndex == classIndex) continue;

            // Create a new dataset structure with only the selected feature
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add((Attribute) syntheticData.attribute(featureIndex).copy());

            // Create a new Instances object with only the selected feature
            Instances singleFeatureData = new Instances("SingleFeatureData", attributes, syntheticData.numInstances());
            singleFeatureData.setClassIndex(-1);

            // Populate with values from the selected feature
            for (int i = 0; i < syntheticData.numInstances(); i++) {
                double[] values = new double[1];
                values[0] = syntheticData.instance(i).value(featureIndex);
                singleFeatureData.add(new DenseInstance(1.0, values));
            }

            // Instantiate VWC with the single-feature dataset
            VWC myV = new VWC(singleFeatureData);
            myV.setDistanceFunction(new EuclideanDistance(singleFeatureData));

            // Calculate beta instances
            int betaInstances;
            if (maxClusterSize == 5) {
                betaInstances = 4; // Special case
            } else {
                betaInstances = Math.max(1, (int)(betaPercentage / 100.0 * singleFeatureData.numInstances()));
            }

            myV.setBeta(betaInstances);
            myV.setMaxClusterSize(maxClusterSize);

            // Perform the clustering
            myV.buildClusterer(singleFeatureData);

            // Replace original values with cluster IDs
            int[] clusterAssignments = myV.getAssignments();
            for (int i = 0; i < syntheticData.numInstances(); i++) {
                syntheticData.instance(i).setValue(featureIndex, clusterAssignments[i]);
            }
        }

        // Reattach the original class attribute if it exists
        if (hasClassAttribute && originalDataWithClass != null) {
            for (int i = 0; i < syntheticData.numInstances(); i++) {
                syntheticData.instance(i).setClassValue(originalDataWithClass.instance(i).classValue());
            }
        }

        return syntheticData;
    }

    /**
     * Normalize the given dataset
     */
    public Instances normalizeData(Instances data) throws Exception {
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        return Filter.useFilter(data, normalize);
    }

    /**
     * Load dataset from file (auto-detects format)
     */
    public static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    /**
     * Save dataset to ARFF file
     */
    public static void saveDataset(Instances data, String filePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(filePath));
        saver.writeBatch();
    }
}
