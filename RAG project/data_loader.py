import os
import pandas as pd

def load_data(directory, word_limit=None):
    data = []

    if os.path.isfile(directory):
        file_path = directory
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                data.append(" ".join(str(value) for value in row if pd.notna(value)))
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data.append(file.read())
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                for _, row in df.iterrows():
                    data.append(" ".join(str(value) for value in row if pd.notna(value)))
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data.append(file.read())

    return data
