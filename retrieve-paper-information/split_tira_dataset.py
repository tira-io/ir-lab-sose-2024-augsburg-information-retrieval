import json
import os


with open('data/tira_documents.json', 'r') as file:
            documents = json.load(file)


chunk_size = 5000

splitted_docs = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]



for i, docs in enumerate(splitted_docs):
    if i < 10:
        with open(f'data/splitted_docs/tira_documents_0{i}.json', 'w') as file:
            json.dump(docs, file)
        print(f'Saved tira_documents_0{i}.json')
    else:
        with open(f'data/splitted_docs/tira_documents_{i}.json', 'w') as file:
            json.dump(docs, file)
        print(f'Saved tira_documents_{i}.json')