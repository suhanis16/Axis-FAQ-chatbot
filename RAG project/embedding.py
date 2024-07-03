import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class Embeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def __call__(self, texts):
        return np.vstack([self.embed_text(text) for text in texts])




