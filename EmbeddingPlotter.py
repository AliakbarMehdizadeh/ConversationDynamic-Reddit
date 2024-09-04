import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingPlotter:
    def __init__(self, embedding_columns):
        """
        Initialize the EmbeddingAnalyzer with the specified embedding columns.
        
        Parameters:
        - embedding_columns: List of column names that contain embedding data
        """
        self.embedding_columns = embedding_columns

    def plot_embedding_trajectories(self, df, ax=None):
        """
        Plot the trajectories of embedding centroids over time.
        
        Parameters:
        - df: DataFrame containing the embeddings and a 'created_time' column
        """
        #time_segments = df['created_time'].unique()

        trajectories = []

        #for time in time_segments:
        for index, row in df.iterrows():
            trajectories.append(row[self.embedding_columns])

        trajectories = np.array(trajectories)
        
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()

        ax.plot(trajectories)
        ax.set_title('Embedding Trajectories Over Time')
        ax.set_xlabel('Time Segments')
        ax.set_ylabel('Embedding Values')
