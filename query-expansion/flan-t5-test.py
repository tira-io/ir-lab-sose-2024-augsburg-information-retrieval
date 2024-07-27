
# Imports
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Create a REST client to the TIRA platform for retrieving the pre-indexed data.
ensure_pyterrier_is_loaded()
tira = Client()

# The dataset: the union of the IR Anthology and the ACL Anthology
# This line creates an IRDSDataset object and registers it under the name provided as an argument.
pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')

# A (pre-built) PyTerrier index loaded from TIRA
index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def generate_expansion(query):
    prompt = (
        "Given the query: '{query}', generate a comprehensive list of keywords that cover the topic. "
        "Include at least 10 keywords or phrases such as synonyms, abbreviations, and related terms. "
        "Ensure the keywords are relevant and useful for improving search or information retrieval. "
        "Format the result as a comma-separated list."
    )
    inputs = tokenizer(prompt.format(query=query), return_tensors="pt")
    outputs = model.generate(**inputs)
    expanded_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return expanded_query

# Initialize the TIRA client and PyTerrier dataset
tira = Client()
pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')

# Example: assuming topics are in a DataFrame
topics_df = pt_dataset.get_topics('text')

# Generate expansions
expanded_queries = []
for index, row in topics_df.iterrows():
    expanded_query = generate_expansion(row['query'])
    expanded_queries.append({"query_id": row['qid'], "query": row['query'], "llm_expansion": expanded_query})

# Save to JSONL file
with open('query-expansion/flan-t5-small-queries_expanded.jsonl', 'w') as f:
    for item in expanded_queries:
        f.write(json.dumps(item) + '\n')
    print("File saved.")

