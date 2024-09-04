import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import product
import networkx as nx

class MarkovModel:
    def __init__(self, data, embedding_columns, n_clusters=4, pca_components=3, order=2):
        """
        Initialize the Markov Model with data, embedding columns, number of clusters, PCA components, and order.

        Parameters:
        - data: DataFrame containing the data
        - embedding_columns: List of columns to use for PCA
        - n_clusters: Number of clusters for K-Means clustering
        - pca_components: Number of PCA components
        - order: Order of mixed states (length of each mixed state tuple)
        """
        self.data = data
        self.embedding_columns = embedding_columns
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.order = order
        self.pca_df = None
        self.transition_matrix = None
        self.mixed_states = None
        self.mixed_state_to_label = None
        self.label_to_idx = None

    def apply_pca(self):
        """
        Apply PCA to reduce dimensionality to the specified number of components.
        """
        df = self.data[self.embedding_columns]
        pca = PCA(n_components=self.pca_components)
        pca_data = pca.fit_transform(df)
        self.pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(self.pca_components)])
    
    def apply_kmeans(self):
        """
        Apply K-Means clustering to discretize the PCA-reduced data into states.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.pca_df['State'] = kmeans.fit_predict(self.pca_df)
    
    def calculate_transition_matrix(self):
        """
        Calculate and normalize the transition matrix based on state transitions.
        """
        transition_matrix = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(1, len(self.pca_df)):
            current_state = self.pca_df['State'].iloc[i-1]
            next_state = self.pca_df['State'].iloc[i]
            transition_matrix[current_state, next_state] += 1
        self.transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    def visualize_transition_matrix(self):
        """
        Visualize the transition matrix as a heatmap.
        """
        plt.figure(figsize=(10, 10), dpi=100)
        sns.heatmap(self.transition_matrix, annot=True, cmap="Blues", fmt=".2f")
        plt.title("Transition Matrix of Conversations on Reddits")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.title(f"Post ID: {self.data.iloc[0]['post_id']}, Subreddit: {self.data.iloc[0]['subreddit']}")
        filename = f"result/plot_post_{self.data.iloc[0]['post_id']}_subreddit_{self.data.iloc[0]['subreddit']}_TransitionMatrix.png"

        plt.savefig(filename)
        plt.close()

    def generate_mixed_states(self):
        """
        Generate all possible mixed states based on the given order from a list of states.
        """
        states = list(self.pca_df['State'].unique())
        self.mixed_states = list(product(states, repeat=self.order))
        mixed_state_labels = [" -> ".join(map(str, mixed_state)) for mixed_state in self.mixed_states]
        self.mixed_state_to_label = {mixed_state: label for mixed_state, label in zip(self.mixed_states, mixed_state_labels)}
        self.label_to_idx = {label: idx for idx, label in enumerate(mixed_state_labels)}

    def calculate_mixed_state_transition_matrix(self):
        """
        Calculate the transition matrix for mixed states.
        """
        n_mixed_states = len(self.mixed_states)
        transition_matrix = np.zeros((n_mixed_states, n_mixed_states))
        for i in range(self.order, len(self.pca_df)-1):
            current_mixed_state = tuple(self.pca_df['State'].values[i-self.order:i])
            next_mixed_state = tuple(self.pca_df['State'].values[i-self.order+1:i+1])
            transition_matrix[self.label_to_idx[self.mixed_state_to_label[current_mixed_state]],
                              self.label_to_idx[self.mixed_state_to_label[next_mixed_state]]] += 1
        self.transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    def visualize_mixed_state_transition_matrix(self):
        """
        Visualize the transition matrix of mixed states as a heatmap.
        """
        mixed_state_labels = list(self.mixed_state_to_label.values())
        plt.figure(figsize=(20, 20), dpi=100)
        sns.heatmap(self.transition_matrix, annot=True, cmap="Blues", fmt=".2f", 
                    xticklabels=mixed_state_labels, yticklabels=mixed_state_labels)
        plt.title("Transition Matrix of Financial Market States with History")
        plt.xlabel("Next Mixed State")
        plt.ylabel("Current Mixed State")
        plt.title(f"Post ID: {self.data.iloc[0]['post_id']}, Subreddit: {self.data.iloc[0]['subreddit']}")
        filename = f"result/plot_post_{self.data.iloc[0]['post_id']}_subreddit_{self.data.iloc[0]['subreddit']}_HigherOrderTMatrix.png"

        plt.savefig(filename)
        plt.close()

    def visualize_network(self):
        """
        Visualize the mixed state transition matrix as a network graph.
        """
        G = nx.DiGraph()
        for mixed_state in self.mixed_states:
            label = self.mixed_state_to_label[mixed_state]
            G.add_node(label)
        for i, current_state in enumerate(self.mixed_states):
            for j, next_state in enumerate(self.mixed_states):
                weight = self.transition_matrix[i, j]
                if weight > 0:
                    G.add_edge(self.mixed_state_to_label[current_state], self.mixed_state_to_label[next_state], weight=weight)
        plt.figure(figsize=(18, 18), dpi=100)
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=500)
        nx.draw_networkx_nodes(G, pos, node_size=5000, node_color="lightblue", alpha=0.8)
        edges = nx.draw_networkx_edges(
            G, pos, arrowstyle="->", arrowsize=20,
            edge_color="black",
            width=[G[u][v]['weight'] * 5 for u, v in G.edges()]
        )
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        plt.title("Markov Model with Mixed States as Network")
        
        plt.title(f"Post ID: {self.data.iloc[0]['post_id']}, Subreddit: {self.data.iloc[0]['subreddit']}")
        filename = f"result/plot_post_{self.data.iloc[0]['post_id']}_subreddit_{self.data.iloc[0]['subreddit']}_Network.png"
        plt.savefig(filename)
        plt.close()
