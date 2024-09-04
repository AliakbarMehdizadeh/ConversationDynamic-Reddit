# Markov Model-Based Reddit Conversations Flow

This project investigates the dynamics of online conversations on Reddit using various techniques such as sentence embedding, PCA, K-Means clustering, and Markov models. By reducing the dimensionality of comment embeddings and clustering them into discrete states, we model transitions between these states using higher-order Markov processes. The project includes visualizations of these state transitions through network graphs, providing a comprehensive view of conversational flow and dynamics over time.

## Features
- **Sentence Embedding**: Convert Reddit comments into dense vector representations.
- **PCA**: Reduce dimensionality of the embedding vectors for easier analysis.
- **K-Means Clustering**: Group the PCA-reduced data into discrete states.
- **Markov Models**: Model transitions between states using higher-order Markov processes.
- **Visualization**: Create network graphs and time series plots to illustrate the dynamics of conversations.


## Usage
1. Clone the repository.
2. Run `pip3 install -r requirements.txt`.
3. Get your credentials from [Reddit](https://www.reddit.com/prefs/apps) and add them to `credentials.py`.
4. Edit `config.py`.
5. Run `python main.py`.

## Output Samples:

### Higher-order network representation of conversation flow incorporating memory:

![Screenshot](result/plot_post_1eta0fd_subreddit_politics_Network.png)

### Dynamics of Sentence Embedding as Conversation unfolds: 

![Screenshot](result/plot_post_1eta0fd_subreddit_politics_angular_magnitude_correlation.png)
![Screenshot](result/plot_post_1eta0fd_subreddit_politics_angular_magnitude_histogram.png)

### Time sereis Representation of Conversation: 

![Screenshot](result/plot_post_1eta0fd_subreddit_politics_angular_magnitude_timeseries.png)



