import json
import networkx as nx
import matplotlib.pyplot as plt


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

    def save_pagerank_matrix(pagerank_matrix: dict, filename: str) -> None:
        """
        Expects a dictionary with the pagerank matrix and a filename.
        Saves the pagerank matrix as a JSON file.
        """
        with open(filename, 'w') as file:
            json.dump(pagerank_matrix, file)


if __name__ == '__main__':
    filename = "pagerank/reference_matrix.json"
    reference_matrix = PageRank.load_reference_matrix(filename)

    G = nx.Graph(reference_matrix)
    
    # PageRank.draw_graph(G)
    # plt.savefig("pagerank/weighted_graph.png")

    pr = nx.pagerank(G, alpha=0.85)

    PageRank.save_pagerank_matrix(pr, "pagerank/pagerank_matrix.json")

    print("Number of edges: ", G.number_of_edges())
    print("Number of nodes: ", G.number_of_nodes())