#%%
##############################
# Description: Script to Define Batch Loader functions

# Import libraries
import pandas as pd
import random
import sys
import pickle as pkl

# Define Vocabulary for users and topics
from torchtext import vocab
from torchtext.data.utils import get_tokenizer
import torch as th

# Import libraries
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# set Device
device = th.device("cuda" if th.cuda.is_available() else "mps")


# Import BertTokenizer
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Import BertModel
Bert = BertModel.from_pretrained('bert-base-uncased')
Bert.to(device)
dim_bert = 768

# Load News
News = pd.read_csv('Data/MINDsmall_train/news.tsv', sep='\t', header=None)
News.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

News_vali = pd.read_csv('Data/MINDsmall_dev/news.tsv', sep='\t', header=None)
News_vali.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

News_con = pd.concat([News, News_vali], ignore_index=True)
News_con = News_con.drop_duplicates(subset=['news_id'])

News_vocab = vocab.build_vocab_from_iterator([[id] for id in  News_con['news_id']], specials=['<unk>'])
News_vocab.set_default_index(News_vocab['<unk>'])
Category_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['category']], specials=['<unk>'])
Category_vocab.set_default_index(Category_vocab['<unk>'])
Subcategory_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['subcategory']], specials=['<unk>'])
Subcategory_vocab.set_default_index(Subcategory_vocab['<unk>'])

#%%
max_title_length = max([len(Tokenizer(title)['input_ids']) for title in News_con['title']])
print("Max title length: ", max_title_length)

# Create News_dict with news_id as key and Category and Subcategory and title as value

title_dict_news = {}
title_token_padded = Tokenizer(News_con.title.values.tolist(),return_tensors='pt',max_length=max_title_length, truncation=True, padding='max_length')
title_token_padded = title_token_padded.to(device)


print("Successfully started News_dict",device,file=sys.stdout)
print("Successfully started News_dict",device,file=sys.stderr)

# %%
with th.no_grad():
    # Create News_dict with news_id as key and Category and Subcategory and title as value
    for idx, (id, Category, SubCategory) in tqdm(enumerate(zip(News_con.news_id, News_con.category, News_con.subcategory))):

        
        bertout = Bert(
            input_ids=title_token_padded['input_ids'][idx].unsqueeze(0),
            attention_mask=title_token_padded['attention_mask'][idx].unsqueeze(0),
            token_type_ids=title_token_padded['token_type_ids'][idx].unsqueeze(0)
        )

        title_dict_news[News_vocab.lookup_indices([id])[0]] = bertout.pooler_output[0]


    


# %%

title_rep = th.zeros((News_vocab.__len__(),dim_bert))

for id in News_con.news_id:
    vocab_id = News_vocab.lookup_indices([id])[0]
    print(vocab_id)
    title_rep[vocab_id] = title_dict_news[vocab_id]

    
# Save title_rep and News_vocab

with open('Bert/title_bert.pkl', 'wb') as f:
    pkl.dump(title_rep, f)

with open('Bert/News_vocab.pkl', 'wb') as f:
    pkl.dump(News_vocab, f)



# %%
