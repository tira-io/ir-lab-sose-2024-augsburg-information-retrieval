#!/usr/bin/env python3
from tira.third_party_integrations import persist_and_normalize_run, ir_datasets, ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import json
import random
from tqdm import tqdm
import pandas as pd
import click
import pyterrier as pt
import re

VALID_PROMPTING_TYPES = {
    'q2d_fs', 'q2d_fs_msmarco', 'q2d_zs', 'q2d_prf',
    'q2e_fs', 'q2e_zs', 'q2e_prf', 'cot', 'cot_prf'
}

def validate_prompting_type(prompting_type):
    if prompting_type not in VALID_PROMPTING_TYPES:
        raise ValueError(f"Invalid prompting type: {prompting_type}. Must be one of {VALID_PROMPTING_TYPES}")

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
        prompt += f"{doc[:1000]}\n"
    prompt += f"Query: {query}\nPassage:"
    return prompt


def q2e_few_shot_prompt(query, examples):
    prompt = "Write a list of keywords for the given query:\n\n"
    for example_query, example_keywords in examples.items():
        prompt += f"Query: {example_query}\n"
        prompt += f"Keywords: {example_keywords}\n\n"
    prompt += f"Query: {query}\nKeywords: "
    return prompt

def q2e_zero_shot_prompt(query):
    prompt = f"Write a list of keywords for the following query: {query}"
    return prompt

def q2e_prf_prompt(query, prf_docs):
    prompt = "Write a list of keywords for the given query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc[:1000]}\n"
    prompt += f"Query: {query}\nKeywords:"
    return prompt

def cot_prompt(query):
    prompt = f"Answer the following query:\n{query}\nGive the rationale before answering."
    return prompt

def cot_prf_prompt(query, prf_docs):
    prompt = "Answer the following query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc[:1000]}\n"
    prompt += f"Query: {query}\n\nGive the rationale before answering."
    return prompt


def retrieve_prf_documents(query, index, docs_store, num_samples=3):
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    prf = bm25.search(query.replace('?', ''))
    # Get the top k docnos
    top_k_docnos = prf['docno'][:num_samples].tolist()
    prf_docs = [docs_store.get(docno).text for docno in top_k_docnos]
    return process_documents(prf_docs)

def process_documents(docs):
    """
    Used for processing documents for prf examples; Getting rid of \n, ABSTRACT, INTRODUCTION, etc.
    """
    processed_docs = []
    for doc in docs:
        # Replace multiple newline characters with a single space
        doc = re.sub(r'\n+', ' ', doc)
        # Remove any periods immediately following "ABSTRACT" or "Abstract"
        doc = re.sub(r'\b(ABSTRACT|Abstract)\s*\.\s*', r'\1 ', doc)
        # Insert a space after "ABSTRACT" or "Abstract" if followed by a non-space character
        doc = re.sub(r'\b(ABSTRACT|Abstract)(\S)', r'\1 \2', doc)
        # Replace occurrences of 'Abstract' or 'ABSTRACT' with a period between words
        doc = re.sub(r'\s*(Abstract|ABSTRACT)\s*', r'. ', doc)
        # Normalize multiple spaces to a single space
        doc = re.sub(r'\s+', ' ', doc)
        # Remove "INTRODUCTION" or "Introduction" followed by a non-space character
        doc = re.sub(r'\b(INTRODUCTION|Introduction)(\S)', r'\2', doc)
        doc = doc.strip()
        processed_docs.append(doc)
    
    return processed_docs


class HuggingFaceTransformerSeq2SeqModel():
    def __init__(self, transformer_model, load_in_8bit):
        #lazy imports: as we now also have a REST api included, it is much faster to not load torch and transformers in case we do not ned it.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model, load_in_8bit=load_in_8bit)
        self._tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    
    def predict(self, prompt):    # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self._device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


class RestApiSeq2SeqModel():
    def __init__(self, rest_api_model):
        from openai import OpenAI as cli
        self._client = cli()
        self._rest_api_model = rest_api_model

    def predict(self, prompt):
        return self._client.chat.completions.create(
            model=self._rest_api_model,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.to_dict()['content']


def generate_expansion(query, seq2seq_model, prompting_type, examples, unique_queries, num_samples=3):

    if prompting_type == "q2d_fs":
        # Randomly select num_samples unique query_texts
        selected_queries = random.sample(unique_queries, num_samples)
        # For each selected query_text, randomly choose one document
        selected_entries = []
        for query_text in selected_queries:
            # Retrieve documents for the current query_text from the preprocessed dictionary
            docs_for_query = examples.get(query_text, [])
            if docs_for_query:
                # Randomly select one document from the list
                doc_text = random.choice(docs_for_query)
                selected_entries.append({"query_text": query_text, "doc_text": doc_text})
            else:
                # Raise an exception if no documents are found for a query_text; should actually not happen
                raise ValueError(f"No documents found for query_text: {query_text}")

        # Format the prompt with the selected examples
        prompt = q2d_few_shot_prompt(query, selected_entries)

    elif prompting_type == "q2d_fs_msmarco":
        # Randomly select num_samples examples for the prompt
        selected_entries = random.sample(examples, num_samples)
        prompt = q2d_few_shot_prompt(query, selected_entries)

    elif prompting_type == "q2d_zs":
        prompt = q2d_zero_shot_prompt(query)

    elif prompting_type == "q2d_prf":
        prompt = q2d_prf_prompt(query, examples)

    elif prompting_type == "q2e_fs":
        # Convert the dictionary to a list of key-value pairs (tuples)
        items = list(examples.items())
        # Randomly select num_samples examples for the prompt
        sampled_pairs = dict(random.sample(items, num_samples))
        prompt = q2e_few_shot_prompt(query, sampled_pairs)

    elif prompting_type == "q2e_zs":
        prompt = q2e_zero_shot_prompt(query)

    elif prompting_type == "q2e_prf":
        prompt = q2e_prf_prompt(query, examples)

    elif prompting_type == "cot":
        prompt = cot_prompt(query)

    elif prompting_type == "cot_prf":
        prompt = cot_prf_prompt(query, examples)

    # Generate the expanded query
    expanded_query = seq2seq_model.predict(prompt)

    return prompt, expanded_query


def save_expanded_queries(expanded_queries, file_path):
    pd.DataFrame(expanded_queries).to_json(file_path, lines=True, orient='records')
    print(f"Expanded queries saved to {file_path}")



@click.command()
@click.option('--prompting-type', default='q2d_fs', help='The type of prompting to use.')
@click.option('--input-dataset', default='ir-lab-sose-2024/ir-acl-anthology-20240504-training', help='The dataset to process.')
@click.option('--seed', default=42, help='The seed for selecting random documents for the context.')
@click.option('--transformer-model', default='google/flan-t5-small', help='The transformer model to use.')
@click.option('--rest-api-model', default=None, help='specify the rest api model, e.g., to use chatgpt by openAI, otherwise, use the transformer-model')
@click.option('--output-dir', default='query-expansion/llm-qe/', help='The output directory.')
def main(input_dataset, transformer_model, prompting_type, seed, output_dir, rest_api_model, load_in_8bit=False):

    # Validate prompting type
    validate_prompting_type(prompting_type)

    # The dataset: the union of the IR Anthology and the ACL Anthology
    dataset = ir_datasets.load(input_dataset)

    # Load the model and tokenizer
    if not rest_api_model:
        seq2seq_model = HuggingFaceTransformerSeq2SeqModel(transformer_model, load_in_8bit)
    else:
        seq2seq_model = RestApiSeq2SeqModel(rest_api_model)
        # to include this into the name.
        transformer_model = rest_api_model
    
    
    examples = None
    unique_queries = None
    # Load query-document or query-keywords examples based on prompting type
    if prompting_type == "q2d_fs":
        with open('query-expansion/query_doc_dict.json', 'r') as file:
            examples = json.load(file)
        # Get the list of unique query_texts from the keys of the dictionary
        unique_queries = list(examples.keys())
    elif prompting_type == "q2d_fs_msmarco":
        with open('query-expansion/ms-marco_query_doc.json', 'r') as file:
            examples = json.load(file)
        unique_queries = None
    elif prompting_type == "q2e_fs":
        with open('query-expansion/query_keywords_dict.json', 'r') as file:
            examples = json.load(file)
        unique_queries = None
    elif prompting_type in ("q2d_prf", "q2e_prf", "cot_prf"):
        # Create a REST client to the TIRA platform for retrieving the pre-indexed data.
        ensure_pyterrier_is_loaded()
        tira = Client()
        index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', input_dataset)
        ####
        docs_store = dataset.docs_store()
        unique_queries = None

    # Generate expansions
    expanded_queries = []
    for query in tqdm(list(dataset.queries_iter())):
        if prompting_type in ("q2d_prf", "q2e_prf", "cot_prf"):
            examples = retrieve_prf_documents(query.default_text(), index, docs_store)
        input_prompt, expanded_query = generate_expansion(query.default_text(), seq2seq_model, prompting_type, examples, unique_queries)
        expanded_queries.append({"query_id": query.query_id, "query":query.default_text(), "llm_expansion": expanded_query, 'llm_prompt': input_prompt})
    
    # Save the expanded queries to a file
    save_expanded_queries(expanded_queries, output_dir + '/' + prompting_type + '-' + transformer_model.replace('/', '-') + '-' + input_dataset.replace('/', '-') + '.jsonl.gz')

if __name__ == "__main__":
    # prompting_type = q2d_fs, q2d_fs_msmarco, q2d_zs, q2d_prf, q2e_fs, q2e_zs, q2e_prf, cot, cot_prf
    main()
