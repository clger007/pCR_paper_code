import pandas as pd
import yaml
import os



def load_config():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dir_name)
    with open("config.yaml", 'r') as config_path:
        try:
            return yaml.safe_load(config_path)
        except yaml.YAMLError as exc:
            print(exc)

config = load_config()

def get_data_info():
    return pd.read_csv(config['cleaned_data_path'], nrows=0).columns.tolist()


def get_data(col=None):
    """
    Due to data security concern, the original note is provided unless the exact col name is specified.
    text: original textual report,
    mask_text: de-identifed report,
    label: is the chart review result for the pcr data
    """
    col = {"text": "text"}.get(col, "mask_text")

    data_df = pd.read_csv(config['cleaned_data_path'])
    df = data_df[[col, 'label']].copy()
    # df = data_df[['summary_3_5_0613', 'label']].copy()
    return df

