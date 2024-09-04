import pandas as pd
import re
from typing import List
from nltk.tokenize import sent_tokenize
import nltk
from typing import List
nltk.download('punkt')  # Download the sentence tokenizer model

class Preprocessor:
    def __init__(self):
        pass
    
    def clean_comment(self, comment: str) -> str:
        """
        Clean the comment text by removing unnecessary characters and whitespace.
        """
        comment = comment.lower()  # Convert to lowercase
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)  # Remove URLs
        # Uncomment if you need to remove mentions, hashtags, and punctuation
        # comment = re.sub(r'@\w+', '', comment)  # Remove mentions
        # comment = re.sub(r'#\w+', '', comment)  # Remove hashtags
        # comment = re.sub(r'[^\w\s]', '', comment)  # Remove punctuation
        comment = comment.strip()  # Remove leading and trailing whitespace
        return comment

    def preprocess_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and split comments into sentences, and add them as new columns in the DataFrame.
        """
        df['cleaned_comment'] = df['comment'].apply(self.clean_comment)
        df['sentences'] = df['cleaned_comment'].apply(lambda x: sent_tokenize(x))
        df = df.dropna(subset=['sentences'])

        # Expand sentences into individual rows
        expanded_rows = []
        for index, row in df.iterrows():
            sentences = row['sentences']
            for i, sentence in enumerate(sentences):
                expanded_row = row.copy()
                expanded_row['sentence'] = sentence
                expanded_row['sentence_index'] = i
                expanded_rows.append(expanded_row)

        # Convert list of expanded rows to DataFrame
        expanded_df = pd.DataFrame(expanded_rows)
        
        # Reset index and use old index as a new column called 'comment_id'
        expanded_df.reset_index(inplace=True)
        expanded_df.rename(columns={'index': 'comment_id'}, inplace=True)

        # Create a MultiIndex
        multi_index = pd.MultiIndex.from_frame(
            expanded_df[['comment_id', 'sentence_index']],
            names=['comment_id', 'sentence_index']
        )
        
        # Set the MultiIndex on the DataFrame
        expanded_df.set_index(multi_index, inplace=True)
        
        # Drop the original columns used to create the MultiIndex if needed
        expanded_df = expanded_df.drop(columns=['comment_id', 'sentence_index'])
        
        return expanded_df
