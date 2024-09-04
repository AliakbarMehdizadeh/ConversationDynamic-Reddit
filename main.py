from credentials import CLIENT_ID, CLIENT_SECRET
from config import TOP_POSTS_TIME_FILTER, MAXIMUM_POSTS_PER_SUBREDDIT, SUBREDDIT_NAMES_LIST, NUM_CLUSTERS, PCA_COMPONENTS
from data_gatherer import DataGatherer
from preprocessing import Preprocessor
from SentenceEmbedder import SentenceEmbeddingAnalyzer
from EmbeddingPlotter import EmbeddingPlotter
from EmbeddingChangeCalculator import EmbeddingChangeCalculator
from EmbeddingDynamics import EmbeddingDynamics
from MarkovModel import MarkovModel

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
sns.set_style("dark")


if __name__ == "__main__":
    data_gatherer = DataGatherer(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        subreddit_names_list=SUBREDDIT_NAMES_LIST,
        top_posts_time_filter=TOP_POSTS_TIME_FILTER,
        maximum_posts_per_subreddit=MAXIMUM_POSTS_PER_SUBREDDIT
    )

    comments_df = data_gatherer.get_comments_list_by_post()
    print('Data retrieval finished.')

    preprocessor = Preprocessor()
    processed_df = preprocessor.preprocess_comments(comments_df)
    print('Data preprocessing finished.')

    print('Sentence embedding started.')
    # Create SentenceEmbeddingAnalyzer instance
    embedding_analyzer = SentenceEmbeddingAnalyzer(
        processed_df=processed_df,
        model_name='bert-base-uncased'
    )

    # Concatenate embeddings with original DataFrame
    comments_df_embedded = embedding_analyzer.add_embeddings()
    print('Embedding process concluded.')

    print('plot embedding as time series')
    embedding_columns = embedding_analyzer._embedding_cols_names
    
    plotter = EmbeddingPlotter(embedding_columns)
    calculator = EmbeddingChangeCalculator(embedding_columns)

    # Group by 'post_id' and 'subreddit'
    grouped_by_post_subreddit = comments_df_embedded.groupby(['post_id', 'subreddit'])

    # Create the result directory if it doesn't exist
    os.makedirs('result', exist_ok=True)  # Safe to use, directory will not be recreated if it already exists

    results = []


    # Plot each group as timeseries
    for (post_id, subreddit), group in grouped_by_post_subreddit:
        print(f'Calculating changes for post_id={post_id}, subreddit={subreddit}')
        result = calculator.calculate_changes(group)
        result['post_id'] = post_id
        result['subreddit'] = subreddit
        results.append(result)

        print(f'Creating plot for post_id={post_id}, subreddit={subreddit}')
        dynamic_analyzer = EmbeddingDynamics(group)
        dynamic_analyzer.plot_magnitude_angular_changes()
        dynamic_analyzer.plot_acf_pacf()
        dynamic_analyzer.plot_histograms()
        # Get statistics
        #stats = dynamic_analyzer.return_stats()

        model = MarkovModel(data=group, embedding_columns=embedding_columns, 
                             n_clusters=NUM_CLUSTERS, pca_components=PCA_COMPONENTS, order=2)
        model.apply_pca()
        model.apply_kmeans()
        model.calculate_transition_matrix()
        model.visualize_transition_matrix()

        model.generate_mixed_states()
        model.calculate_mixed_state_transition_matrix()
        model.visualize_mixed_state_transition_matrix()
        model.visualize_network()


        print(f'Plotting for post_id={post_id}, subreddit={subreddit}')
        
        # Plot the embedding trajectories
        plotter.plot_embedding_trajectories(group)
        # Set the title for the plot
        plt.title(f'Post ID: {post_id}, Subreddit: {subreddit}')
        filename = f'result/plot_post_{post_id}_subreddit_{subreddit}_timeseries.png'
        plt.savefig(filename)
        plt.close()
        


    # Combine all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)


    
