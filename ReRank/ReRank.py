import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

class ReRank: 

    def __init__(self):
        '''Initialize the Rerank class''' 
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def rerank_documents(self, df, top_n):
        '''function to rerank the top n documents in a run df'''
        unique_queries = df['query'].unique()
        reranked_results = []
        
        # Rerank documents for each query
        for query in unique_queries:
            sub_df = df[df['query'] == query]

            # Separate the top top_n documents for reranking
            top_docs = sub_df.nsmallest(top_n, 'rank')
            rest_docs = sub_df.iloc[top_n:]

            documents = top_docs['text'].tolist()

            # Encode the query and documents
            query_embedding = self.encode_text([query])
            document_embeddings = self.encode_text(documents)

            # Compute cosine similarity scores
            cosine_similarity = torch.nn.functional.cosine_similarity
            scores = cosine_similarity(query_embedding, document_embeddings).detach().numpy()
            # Rerank the documents based on similarity scores
            sorted_indices = scores.argsort()[::-1]  # Descending order
            reranked_top_docs = top_docs.iloc[sorted_indices].copy()
            reranked_top_docs['rank'] = range(len(reranked_top_docs))
            reranked_top_docs['score'] = scores[sorted_indices]

            # Concatenate reranked top documents with the rest of the documents
            final_sub_df = pd.concat([reranked_top_docs, rest_docs])

            reranked_results.append(final_sub_df)

        return pd.concat(reranked_results).reset_index(drop=True)


    def encode_text(self, texts):
        '''function to encode queries and documents using BERT'''
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        return embeddings
