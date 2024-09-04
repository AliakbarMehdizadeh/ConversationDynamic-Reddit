import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class EmbeddingDynamics:
    def __init__(self, df, magnitude_column='magnitude_change', angular_column='angular_change'):
        """
        Initializes the DataAnalyzer class with a DataFrame and the specified column names.

        Parameters:
        - df: DataFrame containing the data
        - magnitude_column: Name of the column containing magnitude change values
        - angular_column: Name of the column containing angular change values
        """
        self.df = df
        self.magnitude_column = magnitude_column
        self.angular_column = angular_column
        self.stats= None

    def plot_magnitude_angular_changes(self, ax=None):
        """
        Plots magnitude change and angular change as time series based on row indices.

        Parameters:
        - ax: Matplotlib axis object (optional)
        """
        # Generate a simple time series based on row indices
        times = np.arange(1, len(self.df) + 1)

        plt.figure(figsize=(14, 7))

        # Plot magnitude change
        plt.subplot(2, 1, 1)
        plt.plot(times, self.df[self.magnitude_column], label='Magnitude Change', color='blue')
        plt.title('Magnitude Change Over Time')
        plt.xlabel('Time (row index)')
        plt.ylabel('Magnitude Change')
        plt.grid(True)
        plt.legend()

        # Plot angular change
        plt.subplot(2, 1, 2)
        plt.plot(times, self.df[self.angular_column], label='Angular Change', color='red')
        plt.title('Angular Change Over Time')
        plt.xlabel('Time (row index)')
        plt.ylabel('Angular Change (radians)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        filename = f"result/plot_post_{self.df.iloc[0]['post_id']}_subreddit_{self.df.iloc[0]['subreddit']}_angular_magnitude_timeseries.png"
        plt.savefig(filename)
        plt.close()

    def plot_histograms(self):
        """
        Plots histograms for magnitude change and angular change.
        """
        plt.figure(figsize=(14, 7))

        # Plot histogram for magnitude change
        plt.subplot(2, 1, 1)
        plt.hist(self.df[self.magnitude_column], bins=30, color='blue', alpha=0.7, density=True)
        plt.title('Histogram of Magnitude Change')
        plt.xlabel('Magnitude Change')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Plot histogram for angular change
        plt.subplot(2, 1, 2)
        plt.hist(self.df[self.angular_column], bins=30, color='red', alpha=0.7, density=True)
        plt.title('Histogram of Angular Change')
        plt.xlabel('Angular Change (radians)')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        filename = f"result/plot_post_{self.df.iloc[0]['post_id']}_subreddit_{self.df.iloc[0]['subreddit']}_angular_magnitude_histogram.png"
        plt.savefig(filename)
        plt.close()

    def plot_acf_pacf(self, lags=30):
        """
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
        
        Parameters:
        - time_series: The time series data (array-like)
        - lags: Number of lags to display (default: 40)
        """
        plt.figure(figsize=(14, 6))
    
        # Plot Autocorrelation Function (ACF)
        plt.subplot(221)
        plot_acf(self.df[self.magnitude_column].dropna(), lags=lags, ax=plt.gca())
        plt.title('Autocorrelation Function Magnitude Change')
    
        # Plot Partial Autocorrelation Function (PACF)
        plt.subplot(222)
        plot_pacf(self.df[self.magnitude_column].dropna(), lags=lags, ax=plt.gca())
        plt.title('Partial Autocorrelation Function Magnitude Change')

                # Plot Autocorrelation Function (ACF)
        plt.subplot(223)
        plot_acf(self.df[self.angular_column].dropna(), lags=lags, ax=plt.gca())
        plt.title('Autocorrelation Function Angular Change')
    
        # Plot Partial Autocorrelation Function (PACF)
        plt.subplot(224)
        plot_pacf(self.df[self.angular_column].dropna(), lags=lags, ax=plt.gca())
        plt.title('Partial Autocorrelation Function Angular Change')
    
        plt.tight_layout()
        filename = f"result/plot_post_{self.df.iloc[0]['post_id']}_subreddit_{self.df.iloc[0]['subreddit']}_angular_magnitude_correlation.png"
        plt.savefig(filename)
        plt.close()

    def return_stats(self):
        """
        Computes and returns statistics for magnitude change and angular change.

        Returns:
        - stats: Dictionary containing mean, median, std, skewness, and kurtosis for the specified columns.
        """
        stats = {}

        magnitude_data = self.df[self.magnitude_column].dropna()
        magnitude_mean = magnitude_data.mean()
        magnitude_std = magnitude_data.std()
        magnitude_skew = skew(magnitude_data)
        magnitude_kurtosis = kurtosis(magnitude_data)
        magnitude_median = np.median(magnitude_data)

        stats[self.magnitude_column] = {
            'mean': magnitude_mean,
            'median': magnitude_median,
            'std': magnitude_std,
            'skewness': magnitude_skew,
            'kurtosis': magnitude_kurtosis
        }

        angular_data = self.df[self.angular_column].dropna()
        angular_mean = angular_data.mean()
        angular_std = angular_data.std()
        angular_skew = skew(angular_data)
        angular_kurtosis = kurtosis(angular_data)
        angular_median = np.median(angular_data)

        stats[self.angular_column] = {
            'mean': angular_mean,
            'median': angular_median,
            'std': angular_std,
            'skewness': angular_skew,
            'kurtosis': angular_kurtosis
        }
        self.stats = stats
        return stats        