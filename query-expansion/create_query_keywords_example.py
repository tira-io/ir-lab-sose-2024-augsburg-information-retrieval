# Imports
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pyterrier as pt
import json

def load_data():
    ensure_pyterrier_is_loaded()
    tira = Client()
    pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')
    index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)
    return pt_dataset, index

def perform_query_expansion(index, pt_dataset):
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    # KL-divergence query expansion
    kl = pt.rewrite.KLQueryExpansion(index)
    pipeline_kl = bm25 >> kl
    return pipeline_kl(pt_dataset.get_topics('text'))

def process_query(query):
    # Remove 'applypipeline:off'
    cleaned_query = query.replace('applypipeline:off', '')
    # Split the string by spaces and '^', then take only the first part (the keyword)
    keywords = ' '.join([part.split('^')[0] for part in cleaned_query.split()])
    # Split the keywords into a list
    keyword_list = keywords.split()
    # Limit to the first 20 keywords
    limited_keywords = keyword_list[:20]
    # Join the list back into a comma-separated string
    return ', '.join(limited_keywords)

def create_query_keyword_dict(kl_df):
    kl_df['keywords'] = kl_df['query'].apply(process_query)
    return dict(zip(kl_df['query_0'], kl_df['keywords'])) # 'query_0' is the original query text

def save_to_json(query_keyword_dict, filename):
    with open(filename, 'w') as f:
        json.dump(query_keyword_dict, f, indent=4)

def main():
    pt_dataset, index = load_data()
    kl_df = perform_query_expansion(index, pt_dataset)
    query_keyword_dict = create_query_keyword_dict(kl_df)
    save_to_json(query_keyword_dict, 'query-expansion/query_keywords_dict.json')
    print(query_keyword_dict)

if __name__ == "__main__":
    main()