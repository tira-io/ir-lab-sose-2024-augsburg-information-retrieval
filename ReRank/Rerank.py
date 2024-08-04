import torch
from transformers import BertTokenizer, BertModel

class Rerank: 

    def __init__(self):
        '''Initialize the Rerank class''' 
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def rerankForSingleQuery(self, query, documents): 
        '''function to rank documents given a query'''
        # Encode the query and documents
        query_embedding = self.encode_text([query])
        document_embeddings = self.encode_text(documents)

        # Compute cosine similarity scores
        cosine_similarity = torch.nn.functional.cosine_similarity
        scores = cosine_similarity(query_embedding, document_embeddings)

        # Rerank the documents based on similarity scores
        ranked_indices = scores.argsort(descending=True)
        ranked_documents = [documents[i] for i in ranked_indices]
        return ranked_documents


    def encode_text(self, texts):
        '''function to encode queries and documents using BERT'''
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        return embeddings
