import faiss
import numpy as np
import glob
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

for file in glob.glob("knowledge_base/*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        documents.extend(
            [line.strip() for line in f.readlines() if line.strip()]
        )

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def retrieve_context(query, k=4):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return "\n\n".join([f"- {doc}" for doc in retrieved_docs])
