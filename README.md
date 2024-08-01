# Axis-FAQ-chatbot

## Project Overview

The primary goal of this internship was to develop an advanced banking FAQ chatbot using
Retrieval-Augmented Generation (RAG) techniques and large language models (LLMs). The
project aimed to build a system capable of answering Axis Bank-specific queries while also
handling general questions.
The project is divided into three main components:
1. Knowledge Base Layer
Objective: Given a locally hosted knowledge base, integrate it with a vector database to
store and manage embeddings. This component handles various input formats such as
text files, Excel spreadsheets, and word documents.
2. LLM Layer
Objective: Implement a locally hosted LLM as the question and answer module. The
LLM layer should support multiple LLM options, allowing comparison of their
performance. Additionally, it should support both single-turn Q&A and multi-turn
conversations with context maintenance.
3. Frontend Development
Objective: Develop a simple web-based user interface for the chatbot, enabling users to
interact with the system conveniently