# statistical_analyzer.py
import math
import numpy as np
import matplotlib.pyplot as plt

class StatisticalAnalyzer:
    """
    Performs statistical analysis and visualization:
    - Histogram and boxplot for original and transformed datasets
    - Quartile and IQR calculation
    - Outlier and z-score annotations
    - Custom CDF probability approximation
    """

    @staticmethod
    def generate_dataset(seed=0, size=1000):
        np.random.seed(seed)
        return np.random.gamma(shape=2, scale=20, size=size)

    @staticmethod
    def transform_dataset(data):
        return np.sqrt(data) * np.sin(data / 10) + 20

    @staticmethod
    def calculate_iqr(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        return q1, q3, q3 - q1

    @staticmethod
    def standard_normal_cdf(x):
        # Logistic approximation of standard normal CDF
        return 1 / (1 + math.exp(-1.7 * x))

    @staticmethod
    def analyze_and_visualize():
        original = StatisticalAnalyzer.generate_dataset()
        transformed = StatisticalAnalyzer.transform_dataset(original)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Histogram original
        axs[0, 0].hist(original, bins=50, color='slateblue', alpha=0.7, edgecolor='black')
        axs[0, 0].set_title("Original Dataset Histogram")

        # Boxplot original
        axs[0, 1].boxplot(original, vert=False, patch_artist=True, boxprops=dict(facecolor='lavender'))
        axs[0, 1].set_title("Original Dataset Boxplot")

        # Histogram transformed
        axs[1, 0].hist(transformed, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
        axs[1, 0].set_title("Transformed Dataset Histogram")

        # Boxplot transformed
        axs[1, 1].boxplot(transformed, vert=False, patch_artist=True, boxprops=dict(facecolor='lemonchiffon'))
        axs[1, 1].set_title("Transformed Dataset Boxplot")

        plt.suptitle("Visual Statistical Analyzer")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Numeric summary
        for label, dataset in zip(["Original", "Transformed"], [original, transformed]):
            q1, q3, iqr = StatisticalAnalyzer.calculate_iqr(dataset)
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = dataset[(dataset < lower) | (dataset > upper)]

            print(f"\n--- {label} Dataset Summary ---")
            print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
            print(f"Outlier bounds: < {lower:.2f} or > {upper:.2f}")
            print(f"Outliers detected: {len(outliers)}")

            # Z-score and CDF for mean + 2 std
            mu, sigma = np.mean(dataset), np.std(dataset)
            z_score = 2
            x_val = mu + z_score * sigma
            prob = StatisticalAnalyzer.standard_normal_cdf(z_score)
            print(f"P(X < {x_val:.2f}) ≈ {prob:.4f} using custom CDF")


if __name__ == "__main__":
    StatisticalAnalyzer.analyze_and_visualize()