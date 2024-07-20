import json
import pandas as pd

from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt


def save(dictionary: dict, filename: str) -> None:
    """
    Expects a dictionary with the results and a filename.
    Saves the results as a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(dictionary, file)


if __name__ == '__main__':
    ensure_pyterrier_is_loaded()
    tira = Client()
    pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')
    index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)



    documents_iter = iter(pt_dataset.get_corpus_iter())

    documents = []

    for doc in documents_iter:
        documents.append(doc)

    #documents = pd.DataFrame(documents)

    save(documents, 'data/tira_documents.json')