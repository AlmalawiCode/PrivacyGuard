package privacyguard;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Configuration loader for the synthetic data generation pipeline.
 * Loads settings from config.properties file.
 */
public class ConfigLoader {
    private static Properties properties = new Properties();
    private static boolean loaded = false;
    
    /**
     * Load configuration from config.properties file
     */
    public static void loadConfig() {
        if (loaded) return;
        
        try (InputStream input = new FileInputStream("config.properties")) {
            properties.load(input);
            loaded = true;
            System.out.println("Configuration loaded successfully from config.properties");
        } catch (IOException e) {
            System.err.println("Warning: Could not load config.properties, using default values");
            setDefaultProperties();
            loaded = true;
        }
    }
    
    /**
     * Set default properties if config file is not found
     * These are fallback values that match the current project structure
     */
    private static void setDefaultProperties() {
        // Use relative paths that match current project structure
        properties.setProperty("dataset.bot_iot", "datasets/Original/bot_loT.arff");
        properties.setProperty("dataset.cic_iot", "datasets/Original/CICIoT2023.arff");
        properties.setProperty("dataset.mqtt", "datasets/Original/MQTTset.arff");
        properties.setProperty("dataset.edge_iiot", "datasets/Original/Edge-IIoTset.arff");
        properties.setProperty("output.dir", "output/");
        properties.setProperty("output.synthetic", "output/synthetic/");
        properties.setProperty("output.evaluation", "output/evaluation/");
        properties.setProperty("output.privacy", "output/privacy/");
        properties.setProperty("output.latex", "output/latex/");
        properties.setProperty("privacyguard.s_max", "5");
        properties.setProperty("privacyguard.beta_percent", "5");
        properties.setProperty("vwc.betaPercentage", "5");
        properties.setProperty("vwc.maxClusterSize", "5");
        properties.setProperty("evaluation.train_ratio", "0.7");
        properties.setProperty("evaluation.num_runs", "5");
        properties.setProperty("evaluation.random_seed", "180");
    }
    
    /**
     * Get a property value
     */
    public static String getProperty(String key) {
        if (!loaded) loadConfig();
        return properties.getProperty(key);
    }
    
    /**
     * Get a property value as integer
     */
    public static int getIntProperty(String key) {
        return Integer.parseInt(getProperty(key));
    }
    
    /**
     * Get a property value as double
     */
    public static double getDoubleProperty(String key) {
        return Double.parseDouble(getProperty(key));
    }
    
    /**
     * Get a property value as boolean
     */
    public static boolean getBooleanProperty(String key) {
        return Boolean.parseBoolean(getProperty(key));
    }
    
    /**
     * Get all dataset paths
     */
    public static String[] getDatasetPaths() {
        return new String[] {
            getProperty("dataset.bot_iot"),
            getProperty("dataset.cic_iot"),
            getProperty("dataset.mqtt"),
            getProperty("dataset.edge_iiot")
        };
    }
    
    /**
     * Get synthetic dataset paths for evaluation
     */
    public static String[] getSyntheticDatasetPaths() {
        String evalDir = getProperty("output.evaluation");
        String separator = System.getProperty("file.separator");
        return new String[] {
            evalDir + separator + "bot_loT_train_synthetic_normalized.csv",
            evalDir + separator + "CICIoT2023_train_synthetic_normalized.csv",
            evalDir + separator + "MQTTset_train_synthetic_normalized.csv",
            evalDir + separator + "Edge-IIoTset_train_synthetic_normalized.csv"
        };
    }
}