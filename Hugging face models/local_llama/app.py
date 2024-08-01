from flask import Flask, request, jsonify
from llama2_RAG import *
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    # Process user input and generate bot response
    bot_response = query(QA_LLM, user_input)
    return jsonify({'bot_response': bot_response})

def get_user_input():
    user_input = input("What is your question? \n")
    return user_input


data_path = "D:\Axis-FAQ-chatbot\Data"
documents = load_all_files(data_path)
texts = interpret_files(documents)
embeddings = create_embeddings()

save(texts, embeddings)

model_path = "D:\Axis-FAQ-chatbot\models\llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = load_llm("llama", model_path)
QA_LLM = retrieve_docs(embeddings, llm)


if __name__ == '__main__':
    app.run(debug=True)
