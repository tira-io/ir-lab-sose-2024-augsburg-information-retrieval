import numpy as np

def dict_corpus(corpus):
    return {v[0]: v[1] for v in corpus.values()}

def distance2score(distances):
    """ Euclidean distance transformed to score where highest is best """
    return np.exp(-distances)

def softmax(scores):
    return np.exp(scores) / sum(np.exp(scores))