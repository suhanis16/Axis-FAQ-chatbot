import faiss
from langchain_community.vectorstores import FAISS

def store_embeddings(embeddings, data):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    docstore = {i: doc for i, doc in enumerate(data)}
    index_to_docstore_id = {i: i for i in range(len(data))}
    
    vector_store = FAISS(embedding_function=None, docstore=docstore, index=index, index_to_docstore_id=index_to_docstore_id)
    
    return vector_store

def retrieve_docs(query, vector_store, embedding_model, k=1):
    query_embedding = embedding_model([query])
    D, I = vector_store.index.search(query_embedding, k)
    retrieved_docs = [vector_store.docstore[id] for id in I[0]]
    return retrieved_docs



