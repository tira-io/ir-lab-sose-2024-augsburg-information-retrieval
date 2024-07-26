import json
from collections import defaultdict

def load_papers_info(filename: str) -> dict:
    """
    Expects a filename.
    Loads the papers infos from a JSON file.
    Returns a dictionary with the papers infos.
    """
    with open(filename, 'r') as file:
        papers_info = json.load(file)
    
    return papers_info

def save_reference_matrix(reference_matrix: dict, filename: str) -> None:
    """
    Expects a dictionary with the reference matrix and a filename.
    Saves the reference matrix as a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(reference_matrix, file)

if __name__ == '__main__':
    filename = "data/tira_documents_retrieved.json"

    papers_info = load_papers_info(filename)

    reference_matrix = defaultdict(dict)
    for doc, info in papers_info.items():
        if info is not None:
            for reference in info['references']:
                reference_matrix[info['paperId']][reference['paperId']] = 1
    
    save_reference_matrix(reference_matrix, f'pagerank/reference_matrix.json')
