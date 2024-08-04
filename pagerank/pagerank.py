import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class PageRank:
    def __init__(self):
        pass

    def load_reference_matrix(filename: str) -> dict:
        """
        Expects a filename.
        Loads the reference matrix from a JSON file.
        Returns a dictionary with the reference matrix.
        """
        with open(filename, 'r') as file:
            reference_matrix = json.load(file)
        
        return reference_matrix
    
    def draw_graph(G: nx.graph) -> None:
        nx.draw_circular(G, node_size=400, with_labels=True)

    def save_pageranks(pageranks: dict, filename: str) -> None:
        """
        Expects a dictionary with the pageranks and a filename.
        Saves the pageranks as a JSON file.
        """
        with open(filename, 'w') as file:
            json.dump(pageranks, file)

    def calculate_pageranks(reference_matrix: dict) -> dict:
        """
        Expects a reference matrix.
        Calculates the pageranks.
        Returns the pageranks.
        """
        G = nx.DiGraph(reference_matrix)
        pr = nx.pagerank(G, alpha=0.85)

        print("Number of edges: ", G.number_of_edges())
        print("Number of nodes: ", G.number_of_nodes())
        
        return pr
    
    def save_pub_dates(pub_dates: dict, filename: str) -> None:
        """
        Expects a dictionary with the pub_dates and a filename.
        Saves the pub_dates as a JSON file.
        """
        with open(filename, 'w') as file:
            json.dump(pub_dates, file)


if __name__ == '__main__':

    ########## Create the reference matrix ##########

    # Load the papers info
    with open("data/tira_documents_retrieved.json", 'r') as file:
        papers_info = json.load(file)

    reference_matrix = defaultdict(dict)
    pub_dates = {}

    for doc, info in papers_info.items():
        if info is not None:
            for reference in info['references']:
                reference_matrix[info['paperId']][reference['paperId']] = 1
            
            pub_dates[info['paperId']] = info['publicationDate']

    # Save the reference matrix
    reference_matrix = PageRank.load_reference_matrix("pagerank/reference_matrix.json")
    

    ########## Create the pagerank matrix ##########

    # Load the reference matrix
    with open("pagerank/reference_matrix.json", 'r') as file:
        reference_matrix = json.load(file)
    
    # PageRank.draw_graph(G)
    # plt.savefig("pagerank/weighted_graph.png")

    pr = PageRank.calculate_pageranks(reference_matrix)
    PageRank.save_pageranks(pr, "pagerank/pageranks.json")

    PageRank.save_pub_dates(pub_dates, "pagerank/pub_dates.json")

    print("Successfully calculated the pageranks.")
