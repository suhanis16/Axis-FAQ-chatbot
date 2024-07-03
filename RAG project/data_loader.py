import os
import pandas as pd

def load_data(directory, word_limit=None):
    data = []

    if os.path.isfile(directory): 
        data.extend(load_file(directory, word_limit))
    else: 
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith('.xlsx'):
                data.extend(load_excel(file_path))
            elif filename.endswith('.txt'):
                data.extend(load_text(file_path))

    return data

def load_file(file_path, word_limit=None):
    data = []
    if file_path.endswith('.xlsx'):
        data.extend(load_excel(file_path))
    elif file_path.endswith('.txt'):
        data.extend(load_text(file_path))
    return data

def load_excel(file_path):
    data = []
    df = pd.read_excel(file_path)
    for _, row in df.iterrows():
        data.append(" ".join(str(value) for value in row if pd.notna(value)))
    return data

def load_text(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data.append(file.read())
    return data
