import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

class EmbeddingChangeCalculator:
    def __init__(self, embedding_columns):
        """
        Initialize the EmbeddingChangeCalculator with the specified embedding columns.
        
        Parameters:
        - embedding_columns: List of column names that contain embedding data
        """
        self.embedding_columns = embedding_columns

    def calculate_changes(self, df):
        """
        Calculate the magnitude and angular changes between consecutive embeddings.
        
        Parameters:
        - df: DataFrame containing the embeddings
        
        Returns:
        - DataFrame with added columns for magnitude, magnitude change, and angular change
        """
        # Compute the magnitude (norm) of each embedding
        df['magnitude'] = np.linalg.norm(df[self.embedding_columns].values, axis=1)

        # Compute the magnitude change (difference between consecutive magnitudes)
        df['magnitude_change'] = df['magnitude'].diff()

        # Initialize angular change column
        df['angular_change'] = np.nan

        # Compute the angular change between consecutive embeddings
        for i in range(1, len(df)):
            vec1 = df.loc[df.index[i-1], self.embedding_columns].values
            vec2 = df.loc[df.index[i], self.embedding_columns].values

            # Compute dot product
            dot_product = np.dot(vec1, vec2)
            
            # Compute magnitudes
            magnitude_v1 = np.linalg.norm(vec1)
            magnitude_v2 = np.linalg.norm(vec2)
            
            # Compute cosine of the angle
            cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
            
            # Compute angle in radians
            theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            # Convert to degrees
            theta_degrees = np.degrees(theta_radians)

            df.at[df.index[i], 'angular_change'] = theta_degrees

        return df