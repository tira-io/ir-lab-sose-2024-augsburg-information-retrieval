from tira.third_party_integrations import ir_datasets
import re
import json
import os

def load_data():
    dataset = ir_datasets.load('ir-lab-sose-2024/ir-acl-anthology-20240504-training')
    # Load qrels using qrels_iter
    qrels = list(dataset.qrels_iter())
    # Filter qrels to get only relevant documents (label is 1)
    relevant_qrels = [qrel for qrel in qrels if qrel.relevance == 1]

    docs_store = dataset.docs_store()
    # Get all queries from the dataset
    queries = [i for i in dataset.queries_iter()]
    return relevant_qrels, docs_store, queries

def get_query_text_by_id(queries, query_id):
    for query in queries:
        if query.query_id == query_id:
            return query.query # query text
    return None

def create_query_doc_pairs(relevant_qrels, docs_store, queries):
    query_doc_pairs = []
    for query in relevant_qrels:
        query_id = query.query_id
        query_text = get_query_text_by_id(queries, query_id)
        if query_text:
            doc_id = query.doc_id
            doc_text = docs_store.get(doc_id).text
            query_doc_pairs.append({"query_id": query_id, "query_text": query_text, "doc_id": doc_id, "doc_text": doc_text})
        else:
            print(f"Query text for {query_id} not found.")
    return query_doc_pairs

def process_documents(query_doc_pairs):
    query_doc_dict = {}
    for entry in query_doc_pairs:
        query_text = entry['query_text']
        doc_text = entry['doc_text']
        
        # Replace multiple newline characters with a single space
        doc_text = re.sub(r'\n+', ' ', doc_text)
        # Remove any periods immediately following "ABSTRACT" or "Abstract"
        doc_text = re.sub(r'\b(ABSTRACT|Abstract)\s*\.\s*', r'\1 ', doc_text)
        # Insert a space after "ABSTRACT" or "Abstract" if followed by a non-space character
        doc_text = re.sub(r'\b(ABSTRACT|Abstract)(\S)', r'\1 \2', doc_text)
        # Replace occurrences of 'Abstract' or 'ABSTRACT' with a period between words
        doc_text = re.sub(r'\s*(Abstract|ABSTRACT)\s*', r'. ', doc_text)
        # Normalize multiple spaces to a single space
        doc_text = re.sub(r'\s+', ' ', doc_text)
        # Remove "INTRODUCTION" or "Introduction" followed by a non-space character
        doc_text = re.sub(r'\b(INTRODUCTION|Introduction)(\S)', r'\2', doc_text)
        
        doc_text = doc_text.strip()
        
        if query_text not in query_doc_dict:
            query_doc_dict[query_text] = []
        query_doc_dict[query_text].append(doc_text)
    
    return query_doc_dict

def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    try:
        relevant_qrels, docs_store, queries = load_data()
        query_doc_pairs = create_query_doc_pairs(relevant_qrels, docs_store, queries)
        query_doc_dict = process_documents(query_doc_pairs)
        
        file_path = 'query-expansion/query_doc_dict.json'
        save_to_json(query_doc_dict, file_path)
        print(f"Dictionary saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()