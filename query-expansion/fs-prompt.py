#!/usr/bin/env python3
from tira.third_party_integrations import persist_and_normalize_run, ir_datasets
from tira.rest_api_client import Client
import json
import torch
import random
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import click

device = "cuda" if torch.cuda.is_available() else "cpu"

def q2d_few_shot_prompt(query, examples):
    prompt = "Write a passage that answers the given query:\n\n"
    for example in examples:
        prompt += f"Query: {example['query_text']}\n"
        prompt += f"Passage: {example['doc_text'][:1000]}\n\n"
    prompt += f"Query: {query}\nPassage: "
    return prompt

def q2d_zero_shot_prompt(query):
    prompt = f"Write a passage that answers the following query: {query}"
    return prompt

def q2d_prf_prompt(query, prf_docs):
    prompt = "Write a passage that answers the given query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc}\n"
    prompt += f"Query: {query}\nPassage:"
    return prompt

def q2e_few_shot_prompt(query, examples):
    prompt = "Write a list of keywords for the given query:\n\n"
    for example in examples:
        prompt += f"Query: {example['query_text']}\n"
        prompt += f"Keywords: {example['keywords']}\n\n"
    prompt += f"Query: {query}\nKeywords: "
    return prompt

def q2e_zero_shot_prompt(query):
    prompt = f"Write a list of keywords for the following query: {query}"
    return prompt

def q2e_prf_prompt(query, prf_docs):
    prompt = "Write a list of keywords for the given query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc}\n"
    prompt += f"Query: {query}\nKeywords:"
    return prompt

def cot_prompt(query):
    prompt = f"Answer the following query:\n{query}\nGive the rationale before answering"
    return prompt

def cot_prf_prompt(query, prf_docs):
    prompt = "Answer the following query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc}\n"
    prompt += f"Query: {query}\n\nGive the rationale before answering"
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
    prompt = q2d_few_shot_prompt(query, selected_entries)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate the expanded query
    outputs = model.generate(**inputs, max_new_tokens=200)
    expanded_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return prompt, expanded_query

def save_expanded_queries(expanded_queries, file_path):
    pd.DataFrame(expanded_queries).to_json(file_path, lines=True, orient='records')
    print(f"Expanded queries saved to {file_path}")

@click.command()
@click.option('--input-dataset', default='ir-lab-sose-2024/ir-acl-anthology-20240504-training', help='The dataset to process.')
@click.option('--seed', default=42, help='The seed for selecting random documents for the context.')
@click.option('--transformer-model', default='google/flan-t5-small', help='The transformer model to use.')
@click.option('--output-dir', default='query-expansion/fs-qe-expansions/', help='The output directory.')
def main(input_dataset, transformer_model, seed, output_dir, load_in_8bit=False):
    with open('query-expansion/query_doc_dict.json', 'r') as file:
        query_doc_dict = json.load(file)

    # Get the list of unique query_texts from the keys of the dictionary
    unique_queries = list(query_doc_dict.keys())

    # The dataset: the union of the IR Anthology and the ACL Anthology
    dataset = ir_datasets.load(input_dataset)

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model, load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    # Generate expansions
    expanded_queries = []
    for query in tqdm(list(dataset.queries_iter())):
        input_prompt, expanded_query = generate_expansion(query.default_text(), model, tokenizer, query_doc_dict, unique_queries)
        expanded_queries.append({"query_id": query.query_id, "query":query.default_text(), "llm_expansion": expanded_query, 'llm_prompt': input_prompt})
    
    # Save the expanded queries to a file
    save_expanded_queries(expanded_queries, output_dir + '/' + transformer_model.replace('/', '-') + '-' + input_dataset.replace('/', '-') + '.jsonl.gz')


if __name__ == "__main__":
    main()
