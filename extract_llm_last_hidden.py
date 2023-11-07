import torch
import numpy as np
from data_linkage_and_preprocessing.data_loader import get_data
from transformers import AutoTokenizer, AutoModel


def tokenize_text(row, tokenizer):
    return tokenizer(row["text"], truncation=False, padding=False, return_tensors="pt")

def get_representation(hidden_state, attention_mask, use_mean_pooling):
    if use_mean_pooling:
        return mean_pooling(hidden_state, attention_mask).cpu().numpy()[0]
    else:
        return hidden_state[:, 0].cpu().numpy()[0]

def ceiling_division(n, d):
    return -(n // -d)

def mean_pooling(last_hidden_state, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_hidden_states = torch.sum(last_hidden_state * attention_mask_expanded, 1)
    non_padded_tokens = attention_mask_expanded.sum(1)
    mean_pooled = sum_hidden_states / non_padded_tokens
    return mean_pooled

def handle_short_text(row, model, tokenizer, model_type, max_length, use_mean_pooling, device):

    inputs = tokenizer(row["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        if model_type in ["bert", "bart"]:
            last_hidden_state = model(**inputs).last_hidden_state
        elif model_type == "t5":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            encoder_output = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            last_hidden_state = encoder_output.last_hidden_state
        elif model_type == "gpt":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        return get_representation(last_hidden_state, inputs["attention_mask"], use_mean_pooling)

def handle_long_text(model, tokenizer, 
                        model_type, max_length, 
                        use_mean_pooling,
                        tk_length, full_token_ids, 
                        full_attention_mask, stride, loop, device):
    vals = []
    for i in range(loop):
        start_index = i * stride
        end_index = min(start_index + max_length - 2, tk_length)
        
        if model_type == "bert":
            cls_token_id = tokenizer.cls_token_id
            sep_token_id = tokenizer.sep_token_id
            token_chunk = [cls_token_id] + full_token_ids[start_index:end_index].tolist() + [sep_token_id]
            token_chunk = torch.tensor(token_chunk).to(device).unsqueeze(0)
            attention_mask_chunk = torch.cat([torch.tensor([1], device='cpu'), 
                                                full_attention_mask[start_index:end_index], 
                                                torch.tensor([1], device='cpu')]).unsqueeze(0).to(device)
        elif model_type == "bart":
            if i != 0:
                start_index = start_index - 1  # To account for <s>
            if i != loop - 1:
                end_index = end_index - 1  # To account for </s>
            token_chunk = full_token_ids[start_index:end_index].to(device).unsqueeze(0)
            attention_mask_chunk = full_attention_mask[start_index:end_index].unsqueeze(0).to(device)
        else:
            token_chunk = full_token_ids[start_index:end_index].to(device).unsqueeze(0)
            attention_mask_chunk = full_attention_mask[start_index:end_index].unsqueeze(0).to(device)
        
        with torch.no_grad():
            if model_type in ["bert", "bart"]:
                outputs = model(input_ids=token_chunk, attention_mask=attention_mask_chunk)
                last_hidden_state = outputs.last_hidden_state
            elif model_type == "t5":
                encoder_output = model.encoder(input_ids=token_chunk, attention_mask=attention_mask_chunk)
                last_hidden_state = encoder_output.last_hidden_state
            elif model_type == "gpt":
                outputs = model(input_ids=token_chunk)
                last_hidden_state = outputs.last_hidden_state
            vals.append(get_representation(last_hidden_state, attention_mask_chunk, use_mean_pooling))
    return np.mean(vals, axis=0)

def extract_hidden_states(row, model, tokenizer,
                           model_type, max_length=512, 
                           use_mean_pooling=True, 
                           overlap=100, device='cpu'):
    full_tokenization = tokenize_text(row, tokenizer)
    full_token_ids = full_tokenization["input_ids"][0]
    full_attention_mask = full_tokenization["attention_mask"][0]
    
    tk_length = len(full_token_ids)
    stride = max_length - overlap
    loop = ceiling_division(tk_length - overlap, stride)
    
    if model_type == "bert":
        stride = max_length - overlap - 2
    
    if tk_length <= max_length:
        return handle_short_text(row, 
                                 model, 
                                 tokenizer, 
                                 model_type, 
                                 max_length, 
                                 use_mean_pooling, device)
    else:
        

        return handle_long_text(model, tokenizer, 
                        model_type, max_length, 
                        use_mean_pooling,
                        tk_length, full_token_ids, 
                        full_attention_mask, stride, loop, device)
    

if __name__ == "__main__":

    df = get_data('text')
    d = df.iloc[0]
    bert = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(bert)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(bert).to(device)
    extract_hidden_states(d, model, tokenizer, 't5', max_length=512, use_mean_pooling=True, overlap=100)

