from transformers import pipeline

def generate_answer(question, context, model_name='deepset/roberta-base-squad2'):
    qa_pipeline = pipeline('question-answering', model=model_name)
    result = qa_pipeline(question=question, context=context)
    return result['answer']
