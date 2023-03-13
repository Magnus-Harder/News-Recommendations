#%%
###########################
# Script to fine-tune BERT #
# Data: MSN_news_dataset  #
###########################

##################################
# Hyperparameters: Validation
# Batch_size: 16,32
# Epochs: 2,3,4
# Learning_rate: 5e-5, 3e-5, 2e-5
##################################

# Import libraries
import torch as th
import pandas as pd
import random
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, AdamW
from tqdm import tqdm

# Import BertTokenizer and BertModel
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Model = BertModel.from_pretrained("bert-base-uncased")
#Model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
Model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# Import MSN_news_dataset

News = pd.read_csv('MINDsmall_train/news.tsv', sep='\t', header=None)
News.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
News = News.dropna()

News_vali = pd.read_csv('MINDsmall_dev/news.tsv', sep='\t', header=None)
News_vali.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
News_vali = News_vali.dropna()


# def function to map the sentences to the BertTokenizer
def BertTokenize(title,abstract):
    return Tokenizer(title,abstract,return_tensors='pt')

# Define is Abstract for the title training data
title_train = []
abstract_train = []
label = []

# Create training data
random.seed(1)
for title,abstract in zip(News.title.values,News.abstract.values):
    if random.random() > 0.5:
        title_train.append(title)
        abstract_train.append(abstract)
        label.append(0)
    else:
        index = random.randint(0,len(News.abstract.values)-1)
        title_train.append(title)
        abstract_train.append(News.abstract.values[index])
        label.append(1)


train_tokens = Tokenizer(title_train,abstract_train,return_tensors='pt',max_length=512, truncation=True, padding='max_length')

title_vali = []
abstract_vali = []
label_vali = []

# Create validation data
random.seed(1)
for title,abstract in zip(News_vali.title.values,News_vali.abstract.values):
    title_vali.append(title)
    abstract_vali.append(abstract)

vali_tokens = Tokenizer(title_vali,abstract_vali,return_tensors='pt',max_length=512, truncation=True, padding='max_length')


#train_tokens['labels'] = th.tensor(label).long().reshape(-1,1)



# Masking Training Samples
def mask(N=512, p=0.85):
    return th.distributions.bernoulli.Bernoulli(p).sample((16,N))


#%%
# Create torch Dataset
class NewsData (th.utils.data.Dataset):
    def __init__(self, train_tokens):
        self.train_tokens = train_tokens
        self.len = self.train_tokens['input_ids'].shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {key : val[index] for key,val in self.train_tokens.items()}


Dataset = NewsData(train_tokens)
dataloader = th.utils.data.DataLoader(Dataset, batch_size=8, shuffle=True)

Dataset_vali = NewsData(vali_tokens)
dataloader_vali = th.utils.data.DataLoader(Dataset_vali, batch_size=8, shuffle=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


Model.to(device)
Model.train()

optimizer = th.optim.AdamW(Model.parameters(), lr=5e-6)

loss_fn = th.nn.CrossEntropyLoss()

#%%
epochs = 10

losses_train = []
losses_vali = []

for epoch in range(epochs):
    for batch in tqdm(dataloader, leave= True):
        optimizer.zero_grad()
        batch = {key : val.to(device) for key,val in batch.items()}
        output = Model(**batch)
        loss = loss_fn(th.transpose(output.logits, 1,2), batch['input_ids'])
        loss.backward()
        optimizer.step()
        
        losses_train.append(loss.item())
        

    # Validation
    Model.eval()
    with th.no_grad():
        for batch in tqdm(dataloader_vali, leave= True):
            batch = {key : val.to(device) for key,val in batch.items()}
            output = Model(**batch)
            loss = loss_fn(th.transpose(output.logits,1,2), batch['input_ids'])

            losses_vali.append(loss.item())


# %%

import pickle as pkl

with open('losses_train.pkl', 'wb') as f:
    pkl.dump(losses_train, f)

th.save(Model.state_dict(), 'Bert-Finetune.pt')

# %%
