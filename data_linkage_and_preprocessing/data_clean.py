import pandas as pd
import os
import re
import yaml
import tiktoken
import numpy as np

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens 


def load_data(path: str)-> pd.DataFrame:
    """
    simply load data from scure share drive and return it.
    """
    try:
        df = pd.read_parquet(path)
        df['word_count'] = df['mask_text'].apply(lambda x: x.split(" ").__len__())
        df['tokens'] = df['mask_text'].apply(num_tokens_from_string)
    except:
        wc = np.random.randint(1, 3200, 10)
        d = {
            'id': np.arange(10),
            'word_count': wc ,
            'tokens': wc + 10,
            'masked_text': np.random.randn(10, 128).tolist(),
            'un_masked_text_alert':np.random.randn(10, 128).tolist(),

        }
        print('the data is currently unavailable...')
        df = pd.DataFrame.from_dict(d)
    return df

def clean_data(df):
    """
    the data cleaning procedure involves patients and physicians' information.
    Hence, use the loadData method to directly load de-identified data.
    """
    pass



if __name__ == "__main__":

    with open('data_linkage_and_preprocessing/config.yaml', 'r') as file:
        p = yaml.safe_load(file)
    
    data_path = os.path.join(p['data_path'], p['file_name'])
    df = load_data(data_path)
