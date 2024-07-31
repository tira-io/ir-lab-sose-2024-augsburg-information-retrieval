import pyterrier as pt
import json
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
from prompts import q2d_few_shot_prompt, q2d_zero_shot_prompt, q2d_prf_prompt
from prompts import q2e_few_shot_prompt, q2e_zero_shot_prompt, q2e_prf_prompt
from prompts import cot_prompt, cot_prf_prompt

def generate_expansion(query, model, tokenizer, prompting_type, num_samples=3):
    # prompting_type = q2d_fs, q2d_zs, q2d_prf, q2e_fs, q2e_zs, q2e_prf, cot, cot_prf
    # all should return prompt
    if prompting_type == "q2d_fs":
        pass
    elif prompting_type == "q2d_zs":
        pass
    elif prompting_type == "q2d_prf":
        pass
    elif prompting_type == "q2e_fs":
        pass
    elif prompting_type == "q2e_zs":
        pass
    elif prompting_type == "q2e_prf":
        pass
    elif prompting_type == "cot":
        pass
    elif prompting_type == "cot_prf":
        pass
    else:
        raise ValueError(f"Invalid prompting type: {prompting_type}")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    expanded_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return expanded_query


def save_expanded_queries(expanded_queries, file_path):
    with open(file_path, 'w') as file:
        json.dump(expanded_queries, file, indent=4)
    print(f"Expanded queries saved to {file_path}")

def main(prompting_type):

    ensure_pyterrier_is_loaded()
    tira = Client()

    pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')
    queries = pt_dataset.get_topics('text')

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    expanded_queries = []
    for index, row in queries.iterrows():
        expanded_query = generate_expansion(row['query'], model, tokenizer, prompting_type)
        expanded_queries.append({"query_id": row['qid'], "query": row['query'], "llm_expansion": expanded_query})
    
    save_expanded_queries(expanded_queries, f'query-expansion/flan-t5-xxl_{prompting_type}.jsonl')

if __name__ == "__main__":
    # prompting_type = q2d_fs, q2d_zs, q2d_prf, q2e_fs, q2e_zs, q2e_prf, cot, cot_prf
    main(prompting_type = "q2d_fs")