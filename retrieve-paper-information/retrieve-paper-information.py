import requests
import json
import pandas as pd
from collections import defaultdict
import time
import sys
from tqdm import tqdm


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
    
    if result == {'message': 'Too Many Requests. Please wait and try again or apply for a key for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form', 'code': '429'}:
        print("API rate limit exceeded - waiting 60 seconds!!\n")
        time.sleep(60)
        return retrieve_single_paper_info(title)
    
        # Handle API error
    if not request.ok:
        print("Request is not ok")
        print("title: " + title)
        print("result: ")
        print(result)
        return None
    
    try:
        result = result['data'][0] # Extract the paper information
    except Exception as e:
        print("Unpacking results failed")
        print("title: " + title)
        print("result: ")
        print(result)
        return None

    return result
    

def retrieve_multiple_papers_info(documents: pd.DataFrame) -> dict:
    """
    Expects a dataframe with a 'docno' and 'title' column.
    Returns a dictionary with the docno as key and the paper information (as a dictionary) as value.
    """
    
    docs_with_infos = defaultdict(dict)

    for index, row in tqdm(documents.iterrows(), total=len(documents), unit="Document"):
        docno = row['docno']
        title = row['text'].split('\n')[0]

        docs_with_infos[docno] = retrieve_single_paper_info(title)
    
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
        file_name = f'data/splitted_docs/{console_parameters[0]}'

    print("Started pulling for filename:" + file_name)

    with open(file_name, 'r') as file:
        documents = json.load(file)

    documents = pd.DataFrame(documents)

    start_time = time.time()

    docs_with_infos = retrieve_multiple_papers_info(documents)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {round(execution_time)} seconds")

    save_papers_info(docs_with_infos, f'{file_name}_retrieved.json')

    print("finished:" + file_name)

    # Get the missing results
    # missing_results = [doc for doc, info in docs_with_infos.items() if info is None]
    # print(f'missing results: {round(len(missing_results) / len(documents))}%')
