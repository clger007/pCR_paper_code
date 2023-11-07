import argparse
import logging
from extract_llm_last_hidden import extract_hidden_states
from data_linkage_and_preprocessing.data_loader import get_data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time 
import torch 
import json
import pynvml

tqdm.pandas()

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bart_model_names = [
                    "facebook/bart-base",
                    "facebook/bart-large",
                    "facebook/bart-large-mnli",
                    ]
bert_model_names =  [
                    "distilbert-base-uncased", 
                    "bert-base-uncased",
                    "bert-base-cased", 
                    "emilyalsentzer/Bio_ClinicalBERT",
                    "prajjwal1/bert-small",
                    "prajjwal1/bert-medium",
                    "bert-base-multilingual-cased",
                    "UFNLP/gatortronS",
                    "prajjwal1/bert-tiny"]
T5_models = [      
                    'google/flan-t5-small', 
                    't5-large',
                    't5-small'
                    ]
GPT_models = [
    'gpt2-large', 
    'gpt2'
]

LLMS = {
    'bert': bert_model_names,
    'bart': bart_model_names,
    't5': T5_models,
    'gpt': GPT_models
}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you're using GPU 0
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return meminfo.used / 1024**2  # Convert B to MB

def main(args):
    
    data_type = args.data_type
    data_type_mapping = {"text": "text"}
    data_type = data_type_mapping.get(data_type, "masked_text")



    df = get_data(data_type)
    device = get_device()
    logger.info(f"Using device: {device}")

    extraction_time = {}
    # gpu_consumption = {}
    for model_type, model_names in LLMS.items():
        for model_name in model_names:
            # clear the gpu
            
            initial_gpu_mem = get_gpu_memory()
            
            # start time
            start_time = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            df.loc[:, model_name] = df.progress_apply(lambda x: extract_hidden_states(x, 
                                                                        model, 
                                                                        tokenizer, 
                                                                        model_type, 
                                                                        max_length=512, 
                                                                        use_mean_pooling=True, 
                                                                        overlap=100, device=device), axis=1)
            del model
            torch.cuda.empty_cache()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            final_gpu_mem = get_gpu_memory()
            mem_used = final_gpu_mem - initial_gpu_mem

            extraction_time[model_name] = f"{elapsed_time: .2f} sec"
            # gpu_consumption[model_name] = f"{mem_used: .2f} MB"

    with open(args.execution_time_file, 'w') as exe_time:
        json.dump(extraction_time, exe_time, indent=4)
        # json.dump(gpu_consumption, exe_time, indent=4)
        
    df.to_parquet(args.output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract hidden states from pre-trained LLMs")
    parser.add_argument('-d','--data_type', type=str, default="text", help='Type of data to use.')
    # parser.add_argument('-ss','--sample_size', type=int, default=None, help='Number of samples to process.')
    parser.add_argument('-o','--output_file', type=str, default="result.parquet", help='Output file name.')
    parser.add_argument('-t','--execution_time_file', type=str, default="llm_feature_time.parquet", help='Output exe time file name.')

    args = parser.parse_args()
    
    main(args)