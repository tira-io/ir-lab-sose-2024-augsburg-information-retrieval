
import json
import os

import ir_datasets

def load_corpus(corpus_path, save=True):
    """ Load the document corpus as dict """
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            corpus = json.load(f)
    else:
        dataset = ir_datasets.load("ir-lab-sose-2024/ir-acl-anthology-20240504-training")
        corpus = dataset.docs_store().docs
        if save:
            with open(corpus_path, "w") as f:
                json.dump(obj=corpus, fp=f, indent=2, ensure_ascii=False)
        del dataset # Free space? or is this unnecessary??
    return corpus


def batch_corpus(corpus, batch_size):
    """ Batch generator """
    corpus_keys = list(corpus.keys())
    for anker in range(0, len(corpus), batch_size):
        batch_keys = corpus_keys[anker:anker+batch_size]
        yield {k: corpus[k] for k in batch_keys}


def train_test_split(corpus, split=0.8):
    """ Returns finetuning train-test splits containing only the texts """
    if type(corpus) == str:
        with open(corpus, "r") as f:
            corpus = json.load(f)
    
    split_idx = int(len(corpus)*split)
    train_keys = list(corpus.keys())[:split_idx]
    test_keys = list(corpus.keys())[split_idx:]

    train_texts = [corpus[k][1] for k in train_keys]
    test_texts = [corpus[k][1] for k in test_keys]

    return train_texts, test_texts



import numpy as np

def relevant_corpus(corpus, dataset, n_factor=0):
    """ Get the subset of the corpus of only relevant documents (documents occuring in qrels).
    Optionally you can add a few random nonrelevant documents (by factor n_factor) """
    relevant = list(dataset.get_qrels()["docno"].unique())

    # Some randomly chosen non-relevant documents are included in the corpus subset
    if n_factor > 0:
        nonrelevant = list(corpus.keys() - set(relevant))
        nonrelevant = np.random.choice(nonrelevant, size=int(len(relevant)*n_factor))
        relevant = relevant + list(nonrelevant)

    return {k: corpus[k] for k in relevant}



import torch

def average_pool(last_hidden_states, attention_mask):
    """ Calculates average pooling of hidden states (with attention mask) """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0) 
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode(model, tokenizer, texts, max_length=512, avg_pool=False):
    """ Encode texts with model """
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        if avg_pool: 
            return average_pool(last_hidden_states, inputs["attention_mask"])
        else: # [CLS] embeddings
            return last_hidden_states[:,0,:]