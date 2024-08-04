import faiss
import numpy as np

## NÄCHSTE FRAGE
#def save_index(self, index_dir, comment=""):
#    index_file = self.name + "_".join(list(self.params.values())) + "_" + comment + ".index"
#    faiss.write_index(self.index, os.path.join(index_dir, index_file))

def create_faiss_index(embeddings, index_type, factory_string="", train=False, **index_params):
    """ Create a FAISS index using the index factory. 
    index_type (str): The type of index to create (e.g., "Flat", "HNSW", "IVF", "PQ", etc)
    **index_params: Additional parameters for the index
    Returns the created FAISS index with embeddings added
    """
    embedding_size = embeddings.shape[1]
    
    if not factory_string:
        # Construct the index string
        if index_type == "Flat":
            index_string = f"Flat"
        elif index_type == "HNSW":
            M = index_params.get('M', 16)
            index_string = f"HNSW{M}"
        elif index_type == "LSH":
            #nbits = index_params.get('nbits', embedding_size)
            index_string = "LSH" # nbits not supported with index_factory, will be set to embedding-dim
        elif index_type == "IVF":
            nlist = index_params.get('nlist', 100)
            index_string = f"IVF{nlist},Flat"
        elif index_type == "PQ":
            m = index_params.get('m', 8)
            bits = index_params.get('bits', 8)
            index_string = f"PQ{m}x{bits}"
        elif index_type == "IVFPQ":
            nlist = index_params.get('nlist', 100)
            m = index_params.get('m', 8)
            bits = index_params.get('bits', 8)
            index_string = f"IVF{nlist},PQ{m}x{bits}"
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        factory_string = index_string
    
    # Create the index
    metric = faiss.METRIC_INNER_PRODUCT if index_params.get("metric", "IP") == "IP" else "L2"
    index = faiss.index_factory(embedding_size, factory_string, metric)
    
    # Set additional parameters
    if index_type == "HNSW":
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = index_params.get("efConstruction", 150)
            index.hnsw.efSearch = index_params.get("efSearch", 50)
    elif index_type in ["IVF", "IVFPQ"]:
        index.nprobe = index_params.get("nprobe", 10)
    
    # Normalize embeddings for IP similarity
    if metric == "IP":
        faiss.normalize_L2(embeddings)
    
    if train:
        index.train(embeddings)
    
    index.add(embeddings)
    
    return index



""" EXAMPLES
# Flat index (L2 distance)
print("FLATL2")
flat_l2_index = create_faiss_index(embeddings, "Flat", metric="L2")

print("FLATIP")
# Flat (Inner Product similarity)
flat_ip_index = create_faiss_index(embeddings, "Flat", metric="IP")

print("HNSW")
# HNSW 
hnsw_index = create_faiss_index(embeddings, "HNSW", M=16, efConstruction=100, efSearch=50, metric="L2")

print("IVF")
# IVF index
ivf_index = create_faiss_index(embeddings, "IVF", nlist=100, nprobe=10)

print("PQ")
# PQ (Product Quantization) 
pq_index = create_faiss_index(embeddings, "PQ", m=8, bits=8)

print("LSH")
# LSH (Locality-Sensitive Hashing) 
lsh_index = create_faiss_index(embeddings, "LSH", nbits=128, metric="L2")

print("IVFPQ")
# IVFPQ (IVF with Product Quantization) 
ivfpq_index = create_faiss_index(embeddings, "IVFPQ", nlist=100, m=8, bits=8, nprobe=10)
"""