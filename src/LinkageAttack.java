package privacyguard;

import weka.core.Instance;
import weka.core.Instances;
import java.util.*;

/**
 * Linkage Attack: Simulates an adversary with partial knowledge (quasi-identifiers)
 * attempting to link individuals in the synthetic dataset.
 *
 * This attack assumes the adversary has access to an auxiliary database with some
 * known attributes (quasi-identifiers) and tries to uniquely identify records
 * in the synthetic dataset.
 *
 * @author Abdulmohsen Almalawi <balmalowy@kau.edu.sa>
 */
public class LinkageAttack {

    private Instances originalData;
    private Instances syntheticData;
    private int classIndex;

    // Attack parameters
    private double[] knownAttributeRatios = {0.25, 0.5, 0.75}; // Percentage of attributes known to adversary

    // Results for each knowledge level
    private Map<Double, LinkageResult> results;

    public LinkageAttack(Instances originalData, Instances syntheticData) {
        this.originalData = new Instances(originalData);
        this.syntheticData = new Instances(syntheticData);
        this.classIndex = originalData.classIndex();
        this.results = new LinkedHashMap<>();
    }

    /**
     * Performs linkage attack with varying levels of adversary knowledge
     */
    public void performAttack() {
        System.out.println("\n=== Performing Linkage Attack ===");
        System.out.println("Testing with different adversary knowledge levels...");

        int numNonClassAttributes = originalData.numAttributes() - (classIndex >= 0 ? 1 : 0);

        for (double knownRatio : knownAttributeRatios) {
            int numKnownAttributes = Math.max(1, (int) (knownRatio * numNonClassAttributes));

            System.out.println("\nKnowledge Level: " + (int)(knownRatio * 100) + "% of attributes");
            System.out.println("  Known attributes: " + numKnownAttributes + " out of " + numNonClassAttributes);

            LinkageResult result = performAttackWithKnowledge(numKnownAttributes);
            results.put(knownRatio, result);
        }

        System.out.println("\nLinkage attack completed.");
    }

    /**
     * Perform linkage attack with specific number of known attributes
     */
    private LinkageResult performAttackWithKnowledge(int numKnownAttributes) {
        // Select random attributes as quasi-identifiers (known to adversary)
        List<Integer> quasiIdentifiers = selectQuasiIdentifiers(numKnownAttributes);

        int numRecords = Math.min(originalData.numInstances(), syntheticData.numInstances());
        int uniqueMatches = 0;
        int noMatches = 0;
        int multipleMatches = 0;
        int correctUniqueMatches = 0;

        List<Double> linkabilityScores = new ArrayList<>();

        // For each original record, try to link it to synthetic data using quasi-identifiers
        for (int origIdx = 0; origIdx < numRecords; origIdx++) {
            Instance originalInstance = originalData.instance(origIdx);

            // Find matching synthetic records based on quasi-identifiers
            List<Integer> matchingIndices = findMatchingSyntheticRecords(originalInstance, quasiIdentifiers);

            if (matchingIndices.isEmpty()) {
                noMatches++;
                linkabilityScores.add(0.0); // No linkage possible
            } else if (matchingIndices.size() == 1) {
                uniqueMatches++;
                // Check if it's the correct match (assuming index alignment)
                if (matchingIndices.get(0) == origIdx) {
                    correctUniqueMatches++;
                }
                linkabilityScores.add(1.0); // Unique linkage achieved
            } else {
                multipleMatches++;
                // Partial linkage - narrowed down to a group
                linkabilityScores.add(1.0 / matchingIndices.size());
            }

            if ((origIdx + 1) % 1000 == 0) {
                System.out.println("    Processed " + (origIdx + 1) + "/" + numRecords + " records");
            }
        }

        // Calculate statistics
        double uniqueMatchRate = (double) uniqueMatches / numRecords;
        double correctMatchRate = (double) correctUniqueMatches / numRecords;
        double avgLinkability = linkabilityScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        // Calculate k-anonymity equivalent
        double avgGroupSize = calculateAverageEquivalenceClassSize(quasiIdentifiers);

        return new LinkageResult(
                numKnownAttributes,
                quasiIdentifiers,
                uniqueMatches,
                multipleMatches,
                noMatches,
                correctUniqueMatches,
                uniqueMatchRate,
                correctMatchRate,
                avgLinkability,
                avgGroupSize
        );
    }

    /**
     * Select random attributes as quasi-identifiers (excluding class attribute)
     */
    private List<Integer> selectQuasiIdentifiers(int count) {
        List<Integer> availableAttributes = new ArrayList<>();

        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != classIndex) {
                availableAttributes.add(i);
            }
        }

        // Shuffle and select first 'count' attributes
        Collections.shuffle(availableAttributes, new Random(42)); // Fixed seed for reproducibility

        return availableAttributes.subList(0, Math.min(count, availableAttributes.size()));
    }

    /**
     * Find synthetic records that match the original record on quasi-identifiers
     */
    private List<Integer> findMatchingSyntheticRecords(Instance originalInstance, List<Integer> quasiIdentifiers) {
        List<Integer> matches = new ArrayList<>();

        for (int synIdx = 0; synIdx < syntheticData.numInstances(); synIdx++) {
            Instance syntheticInstance = syntheticData.instance(synIdx);

            boolean allMatch = true;
            for (int attrIdx : quasiIdentifiers) {
                double origValue = originalInstance.value(attrIdx);
                double synValue = syntheticInstance.value(attrIdx);

                // Check if values match (with small tolerance for floating point)
                if (Math.abs(origValue - synValue) > 0.0001) {
                    allMatch = false;
                    break;
                }
            }

            if (allMatch) {
                matches.add(synIdx);
            }
        }

        return matches;
    }

    /**
     * Calculate average equivalence class size (k-anonymity metric)
     */
    private double calculateAverageEquivalenceClassSize(List<Integer> quasiIdentifiers) {
        Map<String, Integer> equivalenceClasses = new HashMap<>();

        // Group synthetic records by quasi-identifier values
        for (int i = 0; i < syntheticData.numInstances(); i++) {
            Instance instance = syntheticData.instance(i);
            StringBuilder key = new StringBuilder();

            for (int attrIdx : quasiIdentifiers) {
                key.append(instance.value(attrIdx)).append("|");
            }

            String keyStr = key.toString();
            equivalenceClasses.put(keyStr, equivalenceClasses.getOrDefault(keyStr, 0) + 1);
        }

        // Calculate average class size
        if (equivalenceClasses.isEmpty()) {
            return 0.0;
        }

        double totalSize = equivalenceClasses.values().stream().mapToInt(Integer::intValue).sum();
        return totalSize / equivalenceClasses.size();
    }

    /**
     * Calculate overall privacy risk score based on linkage attack results
     * Returns 0-1, where 0 is perfect privacy, 1 is no privacy
     */
    public double calculatePrivacyRiskScore() {
        if (results.isEmpty()) {
            throw new IllegalStateException("Must perform attack first");
        }

        // Use the worst-case scenario (highest knowledge level)
        LinkageResult worstCase = results.get(knownAttributeRatios[knownAttributeRatios.length - 1]);

        // Risk is primarily based on unique match rate
        double uniqueMatchRisk = worstCase.uniqueMatchRate;

        // Also consider average linkability
        double linkabilityRisk = worstCase.avgLinkability;

        // Weighted combination
        return (0.7 * uniqueMatchRisk) + (0.3 * linkabilityRisk);
    }

    /**
     * Print detailed report of linkage attack results
     */
    public void printReport() {
        System.out.println("\n=== LINKAGE ATTACK REPORT ===");
        System.out.println("Dataset: " + originalData.relationName());
        System.out.println("Number of records: " + syntheticData.numInstances());
        System.out.println("Number of attributes: " + (syntheticData.numAttributes() - (classIndex >= 0 ? 1 : 0)));
        System.out.println();

        for (Map.Entry<Double, LinkageResult> entry : results.entrySet()) {
            double knowledgePercent = entry.getKey() * 100;
            LinkageResult result = entry.getValue();

            System.out.println("--- Adversary Knowledge Level: " + (int)knowledgePercent + "% ---");
            System.out.println("  Known attributes (quasi-identifiers): " + result.numKnownAttributes);
            System.out.println();

            System.out.printf("  Unique matches:          %6d (%.2f%%)\n",
                    result.uniqueMatches,
                    result.uniqueMatchRate * 100);

            System.out.printf("  Multiple matches:        %6d (%.2f%%)\n",
                    result.multipleMatches,
                    (double)result.multipleMatches / syntheticData.numInstances() * 100);

            System.out.printf("  No matches:              %6d (%.2f%%)\n",
                    result.noMatches,
                    (double)result.noMatches / syntheticData.numInstances() * 100);

            System.out.printf("  Correct unique matches:  %6d (%.2f%%)\n",
                    result.correctUniqueMatches,
                    result.correctMatchRate * 100);

            System.out.println();
            System.out.printf("  Average linkability score:     %.4f\n", result.avgLinkability);
            System.out.printf("  Average equivalence class size: %.2f (k-anonymity metric)\n", result.avgGroupSize);
            System.out.println();
        }

        System.out.println("=== OVERALL PRIVACY ASSESSMENT ===");
        double riskScore = calculatePrivacyRiskScore();
        System.out.printf("Privacy Risk Score: %.4f (0=perfect, 1=no privacy)\n", riskScore);
        System.out.println();

        if (riskScore < 0.1) {
            System.out.println("Privacy Level: EXCELLENT (Very Low Risk)");
            System.out.println("The synthetic data strongly resists linkage attacks.");
            System.out.println("Even with 75% attribute knowledge, adversary cannot reliably link records.");
        } else if (riskScore < 0.3) {
            System.out.println("Privacy Level: GOOD (Low Risk)");
            System.out.println("The synthetic data provides good protection against linkage attacks.");
            System.out.println("Linkage is difficult even with significant attribute knowledge.");
        } else if (riskScore < 0.5) {
            System.out.println("Privacy Level: MODERATE (Medium Risk)");
            System.out.println("The synthetic data provides moderate protection against linkage attacks.");
            System.out.println("Adversaries with high attribute knowledge may achieve some linkage.");
        } else if (riskScore < 0.7) {
            System.out.println("Privacy Level: POOR (High Risk)");
            System.out.println("The synthetic data is vulnerable to linkage attacks.");
            System.out.println("Adversaries can successfully link many records with partial knowledge.");
        } else {
            System.out.println("Privacy Level: VERY POOR (Very High Risk)");
            System.out.println("The synthetic data provides minimal protection against linkage attacks.");
            System.out.println("Records can be easily linked even with limited knowledge.");
        }

        System.out.println("\n" + "=".repeat(50));
    }

    /**
     * Get detailed statistics for export
     */
    public Map<String, Object> getDetailedStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();

        for (Map.Entry<Double, LinkageResult> entry : results.entrySet()) {
            double knowledgePercent = entry.getKey() * 100;
            LinkageResult result = entry.getValue();
            String prefix = "Knowledge_" + (int)knowledgePercent + "%_";

            stats.put(prefix + "Unique_Match_Rate", result.uniqueMatchRate);
            stats.put(prefix + "Correct_Match_Rate", result.correctMatchRate);
            stats.put(prefix + "Avg_Linkability", result.avgLinkability);
            stats.put(prefix + "Avg_Group_Size", result.avgGroupSize);
        }

        stats.put("Overall_Privacy_Risk_Score", calculatePrivacyRiskScore());

        return stats;
    }

    /**
     * Inner class to store linkage attack results
     */
    private static class LinkageResult {
        int numKnownAttributes;
        List<Integer> quasiIdentifiers;
        int uniqueMatches;
        int multipleMatches;
        int noMatches;
        int correctUniqueMatches;
        double uniqueMatchRate;
        double correctMatchRate;
        double avgLinkability;
        double avgGroupSize;

        LinkageResult(int numKnownAttributes, List<Integer> quasiIdentifiers,
                      int uniqueMatches, int multipleMatches, int noMatches,
                      int correctUniqueMatches, double uniqueMatchRate,
                      double correctMatchRate, double avgLinkability, double avgGroupSize) {
            this.numKnownAttributes = numKnownAttributes;
            this.quasiIdentifiers = quasiIdentifiers;
            this.uniqueMatches = uniqueMatches;
            this.multipleMatches = multipleMatches;
            this.noMatches = noMatches;
            this.correctUniqueMatches = correctUniqueMatches;
            this.uniqueMatchRate = uniqueMatchRate;
            this.correctMatchRate = correctMatchRate;
            this.avgLinkability = avgLinkability;
            this.avgGroupSize = avgGroupSize;
        }
    }
}
