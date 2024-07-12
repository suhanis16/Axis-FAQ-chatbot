import gc
import streamlit as st
import warnings
import torch
import pandas as pd
import pathlib
import docx
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

warnings.filterwarnings("ignore")

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def load_all_files(directory_path):
    data = []
    for file_path in pathlib.Path(directory_path).glob("*"):
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = " ".join(str(value) for value in row.values)
                data.append(Document(page_content=content))
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data.append(Document(page_content=content))
        elif file_path.suffix == '.docx':
            content = read_docx(file_path)
            data.append(Document(page_content=content))
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                content = " ".join(str(value) for value in row.values)
                data.append(Document(page_content=content))
    return data

def interpret_files(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    return embeddings

def save(texts, embeddings):
    db=Chroma.from_documents(texts,embedding=embeddings,persist_directory="test_index")
    db.persist()

def load_llm(model_name):
    if model_name == "phi3":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True,)
    elif model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True,)
    elif model_name == "gemma":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True,)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def retrieve_docs(embeddings, llm):
    vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings)
    retriever = vectordb.as_retriever(search_kwargs = {"k" : 2})

    qna_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
       template=qna_prompt_template, input_variables=["context","question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=False,
                                        chain_type_kwargs={'prompt': PROMPT})
    return chain

def answer_question(chain, question):
    time_start = time.time()
    output = chain({'query': question})
    response = output["result"]
    time_elapsed = time.time() - time_start
    return response, time_elapsed

def main():
    st.title("Banking FAQ Chatbot")

    # Load documents and create embeddings
    data_path = st.text_input("Enter the path to the data directory", "/kaggle/input/axisfaq-data")
    if st.button("Load Data"):
        with st.spinner("Loading documents..."):
            documents = load_all_files(data_path)
            texts = interpret_files(documents)
            embeddings = create_embeddings()
            save(texts, embeddings)
        st.success("Documents loaded and embeddings created!")

    model_name = st.selectbox("Select the model", ["phi3", "llama", "gemma"])
    if st.button("Load Model"):
        with st.spinner(f"Loading {model_name} model..."):
            llm = load_llm(model_name)
            QA_LLM = retrieve_docs(embeddings, llm)
        st.success(f"{model_name} model loaded!")

    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            answer, time_elapsed = answer_question(QA_LLM, question)
        st.write(f"**Answer:** {answer}")
        st.write(f"*Response time:* {time_elapsed:.02f} sec")

if __name__ == "__main__":
    main()
