import requests
import json
import pandas as pd
from collections import defaultdict
import time

from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt
import sys



def retrieve_single_paper_info(title: str) -> dict:
    """
    Expects a title.
    Queries the Semantic Scholar API for the paper information.
    Returns a dictionary with the paper information.
    """

    # Query the Semantic Scholar API for the paper information
    request = requests.get(
    f'https://api.semanticscholar.org/graph/v1/paper/search/match?query={title}',
    params={'fields': 'title,abstract,publicationDate,referenceCount,citationCount,references'}
    )  
    result = request.json()

    if result == {'error': 'Title match not found'}:
        return None
    
    if not request.ok:
        raise Exception(f'Request failed: {request.reason}')

    result = result['data'][0] # Extract the paper information

    return result
    

def retrieve_multiple_papers_info(documents: pd.DataFrame) -> dict:
    """
    Expects a dataframe with a 'docno' and 'title' column.
    Returns a dictionary with the docno as key and the paper information (as a dictionary) as value.
    """
    
    docs_with_infos = defaultdict(dict)
    iteration = 0

    for index, row in documents.iterrows():
        docno = row['docno']
        title = row['text'].split('\n')[0]

        docs_with_infos[docno] = retrieve_single_paper_info(title)

        if(docs_with_infos[docno] is not None):
            print(title)
            print(docs_with_infos[docno]['title'])
    

        iteration += 1
        if iteration % 100 == 0:
            print(f'Processed {iteration} documents')
            break
    
    return docs_with_infos


def save_papers_info(papers_info: dict, filename: str) -> None:
    """
    Expects a dictionary with the infos on the papers and a filename.
    Saves the papers infos as a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(papers_info, file)


def load_papers_info(filename: str) -> dict:
    """
    Expects a filename.
    Loads the papers infos from a JSON file.
    Returns a dictionary with the papers infos.
    """
    with open(filename, 'r') as file:
        papers_info = json.load(file)
    
    return papers_info



if __name__ == '__main__':
    console_parameters = sys.argv[1:] # get the parameters from the console

    if len(console_parameters) == 0:
        file_name = 'data/tira_documents.json'
    else:
        file_name = f'data/splitted_docs/{console_parameters[0]}.json'
    
    with open(file_name, 'r') as file:
        documents = json.load(file)
    
    documents = pd.DataFrame(documents)

    start_time = time.time()

    docs_with_infos = retrieve_multiple_papers_info(documents)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {round(execution_time)} seconds")

    save_papers_info(docs_with_infos, file_name)

    # Get the missing results
    # missing_results = [doc for doc, info in docs_with_infos.items() if info is None]
    # print(f'missing results: {round(len(missing_results) / len(documents))}%')
