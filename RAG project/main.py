import os
from dotenv import load_dotenv
from data_loader import load_data
from embedding import Embeddings
from vector_db import store_embeddings, retrieve_docs
from qa_model import generate_answer

def main(directory, query, word_limit=None):

    data = load_data(directory, word_limit)
    
    embedding_model = Embeddings()
    embeddings = embedding_model(data)
    
    
    vector_store = store_embeddings(embeddings, data)

    
    similar_documents = retrieve_docs(query, vector_store, embedding_model)
 
    context = ' '.join(similar_documents)
    print(f"Answer: {context}")
    
    answer = generate_answer(query, context)
    print(f"Answer generated: {answer}")  # Print the generated answer
    
    return answer, similar_documents

if __name__ == "__main__":
    load_dotenv()
    directory = input("Enter the path to your directory or file: ")
    query = input("Enter your query: ")
    
    answer, similar_documents = main(directory, query)
    print(f"Answer: {answer}")  # Print the answer obtained from main function
    for i, doc in enumerate(similar_documents):
        print(f"Document {i+1}: {doc}")

