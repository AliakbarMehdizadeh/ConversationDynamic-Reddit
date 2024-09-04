import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from typing import List

class SentenceEmbeddingAnalyzer:
    def __init__(self, processed_df: pd.DataFrame, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self._embedding_cols_names = None
        self.embeddings_df = None
        self.processed_df = processed_df
    
    def _get_embedding(self, sentence: str) -> np.ndarray:
        """
        Get BERT embeddings for a single sentence.
        """
        inputs = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the embedding for the [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    def analyze_embeddings(self, sentences: List[str]) -> pd.DataFrame:
        """
        Analyze embeddings for a list of sentences.
        """
        embeddings = np.array([self._get_embedding(sentence) for sentence in sentences])
        # Create column names as embedding_0, embedding_1, ...
        num_features = embeddings.shape[1]
        column_names = [f'embedding_{i}' for i in range(num_features)]
        df = pd.DataFrame(embeddings, columns=column_names)
        self._embedding_cols_names = column_names  # Store column names for later use
        df.index = self.processed_df.index
        return df

    def add_embeddings(self) -> pd.DataFrame:
        """
        Add embeddings to the processed DataFrame.
        """
        self.embeddings_df = self.analyze_embeddings(self.processed_df['sentence'])
        return pd.concat([self.processed_df, self.embeddings_df], axis=1)

    @property
    def embedding_cols_names(self) -> List[str]:
        if self._embedding_cols_names is None:
            raise ValueError("Embedding columns names are not available. Run 'analyze_embeddings' first.")
        return self._embedding_cols_names
