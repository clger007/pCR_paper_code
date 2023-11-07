from tqdm.notebook import tqdm
import io
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import re
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from transformers import (GPT2Tokenizer,
                          GPT2Model, 
                          get_linear_schedule_with_warmup,
                          )
from data_linkage_and_preprocessing.data_loader import get_data
from sklearn.model_selection import train_test_split
import numpy as np



class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.gpt = model
        self.dropout = nn.Dropout(0.05)
        # self.fc1 = nn.Linear(1024, 512)
        self.rnn = nn.GRU(1024, 512, batch_first=False)
        self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):

        gpt2_output = self.gpt(sent_id, attention_mask=mask)
        # mean_pooling_rep = mean_pooling(gpt2_output[0], mask)
        # mean_pooling_rep = self.dropout(mean_pooling_rep)
        rnn_output, _  = self.rnn(gpt2_output['last_hidden_state'].squeeze(0))
        final_hidden_rnn = rnn_output[0, -1, :]
        x = self.fc1(final_hidden_rnn)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
def resample(method, df, label):
    neg_df = df[df[label] == 0]
    pos_df = df[df[label] == 1]
    
    if method == 'over':
        pos_df = pos_df.sample(len(neg_df), replace=True)
    else:
        neg_df = neg_df.sample(len(pos_df))

    merged = pd.concat([neg_df, pos_df])
    merged.reset_index(inplace=True)
    return merged 

def tokenizer_plus(text, tokenizer, max_len=1024):
    white_removed = re.sub(r"\s+", " ", text)
    res = tokenizer(white_removed, return_tensors='pt')
    tk_len = res['attention_mask'].sum()

    if tk_len > max_len + max_len//2:

        number_of_chunks = tk_len//max_len
        if tk_len%max_len > max_len//2:
            number_of_chunks += 1
        
        res_new = tokenizer(white_removed, max_length=number_of_chunks * max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = res_new['input_ids'].reshape(1, number_of_chunks, max_len)
        attention_mask = res_new['attention_mask'].reshape(1, number_of_chunks, max_len)
        return input_ids, attention_mask
    else:
        res_new = tokenizer(white_removed, max_length=max_len, padding='max_length', return_tensors='pt')
        return res['input_ids'], res['attention_mask']
    
def mean_pooling(last_hidden_state, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_hidden_states = torch.sum(last_hidden_state * attention_mask_expanded, 1)
    non_padded_tokens = attention_mask_expanded.sum(1)
    mean_pooled = sum_hidden_states / non_padded_tokens
    return mean_pooled

data_df = get_data('text')
data_df.columns = ['text', 'pcr']
gpt_model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2Model.from_pretrained(gpt_model_name)
data_df['tk_count'] = data_df['text'].apply(lambda x: tokenizer(x, return_tensors='pt')['input_ids'][0].shape[0])
data_df['pcr'] = data_df['pcr'].apply(lambda x: 1 if x=='Yes' else 0)


for param in model.parameters():
    param.requires_grad = False

clf = Classifier(model)

optimizer = optim.AdamW(clf.parameters(), lr=3e-4)
cross_entropy = nn.NLLLoss()


non_test_df, test_df = train_test_split(data_df, stratify=data_df['pcr'], test_size=.2)
train_df, dev_df = train_test_split(non_test_df, stratify=non_test_df['pcr'], test_size=.1)

num_epochs = 100


num_epochs = 20
device = 'cuda'

clf.to(device)


for epoch in range(num_epochs):
    clf.train()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    total_preds = []
    total_labels = []

    running_loss = 0.0
    # iterate over batches

    for index, row in train_df.iterrows():
        tokens, masks = tokenizer_plus(row['text'], tokenizer, max_len=1024)
        label = torch.tensor([row["pcr"]], dtype=torch.long).to(device)
        optimizer.zero_grad()
        tokens = tokens.to(device)
        masks = masks.to(device)
        output = clf(tokens, masks)
        loss = cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_df)}")

    clf.eval()
    val_loss = 0


    # compute the training loss of the epoch
    with torch.no_grad():
        for index, row in dev_df.iterrows():
            tokens, mask = tokenizer_plus(row["text"], tokenizer, max_len=1024)
            tokens = tokens.to(device)
            masks = masks.to(device)
            label = torch.tensor([row["pcr"]], dtype=torch.long).to(device)
            output = clf(tokens, mask)
            loss = cross_entropy(output, label)
            val_loss += loss.item()
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(dev_df)}")
    torch.save(model.state_dict(), f'E:/Project/pCR/classification/models/other_{epoch}_lora.pt')
