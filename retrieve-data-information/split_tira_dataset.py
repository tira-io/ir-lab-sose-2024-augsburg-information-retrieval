import json


with open('tira_documents.json', 'r') as file:
            documents = json.load(file)


print(type(documents))



chunk_size = 1000

splitted_docs = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]



for i, docs in enumerate(splitted_docs):
    with open(f'data/splitted_docs/tira_documents_{i}.json', 'w') as file:
        json.dump(docs, file)
    print(f'Saved tira_documents_{i}.json')