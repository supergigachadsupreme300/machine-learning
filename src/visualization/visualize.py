"""
Visualization utilities for exploratory data analysis and results
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(data, title="Distribution"):
    """
    Plot distribution of data
    
    Args:
        data: Array-like data to plot
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.show()


def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot correlation matrix heatmap
    
    Args:
        df: DataFrame
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded")
