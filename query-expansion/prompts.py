def q2d_few_shot_prompt(query, examples):
    prompt = "Write a passage that answers the given query:\n\n"
    for example in examples:
        prompt += f"Query: {example['query_text']}\n"
        prompt += f"Passage: {example['doc_text']}\n\n"
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
    prompt = f"Answer the following query:\n{query}\nGive the rationale before answering."
    return prompt

def cot_prf_prompt(query, prf_docs):
    prompt = "Answer the following query based on the context:\n\nContext: "
    for doc in prf_docs:
        prompt += f"{doc}\n"
    prompt += f"Query: {query}\n\nGive the rationale before answering."
    return prompt