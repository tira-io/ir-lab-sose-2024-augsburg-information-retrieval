import json
import os

def load_papers_info(filename: str) -> dict:
    """
    Expects a filename.
    Loads the papers infos from a JSON file.
    Returns a dictionary with the papers infos.
    """
    with open(filename, 'r') as file:
        papers_info = json.load(file)
    
    return papers_info

def get_splitted_docfiles(directory_path: str) -> list:
    try:
        files = os.listdir(directory_path)
        return files
    except Exception as e:
        print(f"Error reading directory {directory_path}: {e}")
        return []

def save_papers_info(papers_info: dict, filename: str) -> None:
    """
    Expects a dictionary with the infos on the papers and a filename.
    Saves the papers infos as a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(papers_info, file)

if __name__ == '__main__':
    directory_path = "data/splitted_docs/"
    files = get_splitted_docfiles(directory_path)

    papers_info = {}
    for file in files:
        papers_info.update(load_papers_info(directory_path + file))
    
    save_papers_info(papers_info, 'data/tira_documents_retrieved.json')

    print(len(papers_info))

    # Get the missing results
    missing_results = [doc for doc, info in papers_info.items() if info is None]
    print(f'Percentage of missing results: {round((len(missing_results) / len(papers_info))*100)}%')

    print("Successfully combined the splitted documents.")
