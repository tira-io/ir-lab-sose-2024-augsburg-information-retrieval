from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt
import json
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def format_prompt(examples, query):
    prompt = "Write a passage that answers the given query:\n\n"
    for example in examples:
        prompt += f"Query: {example['query_text']}\n"
        prompt += f"Passage: {example['doc_text']}\n\n"
    prompt += f"Query: {query}\nPassage:"
    return prompt

def generate_expansion(query, model, tokenizer, query_doc_dict, unique_queries, num_samples=3):
    # Randomly select num_samples unique query_texts
    selected_queries = random.sample(unique_queries, num_samples)
    # For each selected query_text, randomly choose one document
    selected_entries = []
    for query_text in selected_queries:
        # Retrieve documents for the current query_text from the preprocessed dictionary
        docs_for_query = query_doc_dict.get(query_text, [])
        if docs_for_query:
            # Randomly select one document from the list
            doc_text = random.choice(docs_for_query)
            selected_entries.append({"query_text": query_text, "doc_text": doc_text})
        else:
            # Raise an exception if no documents are found for a query_text; should actually not happen
            raise ValueError(f"No documents found for query_text: {query_text}")

    # Format the prompt with the selected examples
    prompt = format_prompt(selected_entries, query)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate the expanded query
    outputs = model.generate(**inputs, max_new_tokens=200)
    expanded_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return expanded_query

def save_expanded_queries(expanded_queries, file_path):
    with open(file_path, 'w') as file:
        json.dump(expanded_queries, file, indent=4)
    print(f"Expanded queries saved to {file_path}")

def main():

    # Load the dictionary from the JSON file
    with open('query-expansion/query_doc_dict.json', 'r') as file:
        query_doc_dict = json.load(file)

    # Get the list of unique query_texts from the keys of the dictionary
    unique_queries = list(query_doc_dict.keys())

    # Create a REST client to the TIRA platform for retrieving the pre-indexed data.
    ensure_pyterrier_is_loaded()
    tira = Client()

    # The dataset: the union of the IR Anthology and the ACL Anthology
    # This line creates an IRDSDataset object and registers it under the name provided as an argument.
    pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')

    queries = pt_dataset.get_topics('text')

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    # Generate expansions
    expanded_queries = []
    for index, row in queries.iterrows():
        expanded_query = generate_expansion(row['query'], model, tokenizer, query_doc_dict, unique_queries)
        expanded_queries.append({"query_id": row['qid'], "query": row['query'], "llm_expansion": expanded_query})
    
    # Save the expanded queries to a file
    save_expanded_queries(expanded_queries, 'query-expansion/fs_qe_flan-t5-small.jsonl')


if __name__ == "__main__":
    main()