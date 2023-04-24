#%%
# Import pytorch elements
import torch as th
from torch.utils.data import Dataset, DataLoader


# Import text utils
from nltk.tokenize import RegexpTokenizer

# Import basic packages
import pandas as pd
import numpy as np
import pickle

# import GloVe

# import word_dict
with open('Data/MINDdemo_utils/word_dict_all.pkl', 'rb') as f:
    word_dict = pickle.load(f)


class NewsDataset(Dataset):
    def __init__(self, user_file, news_file, word_dict_file, max_history_length=50, max_title_length=20, max_abstract_length=100,userid_dict = None, mask_prob= 0.5,device='cpu'):

        # Define Parameters        
        self.device = device
        self.mask_prob = mask_prob
        self.max_history_length = max_history_length
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length

        # Get user_id_dict
        if userid_dict is None:
            self.userid_dict = {user_id: idx+1 for idx, user_id in enumerate(self.user_data['user_id'].unique())}
        else:
            self.userid_dict = userid_dict

        # load_word_dict
        with open(word_dict_file, 'rb') as f:
            self.word_dict = pickle.load(f)

        # Define tokenizer and encoder functions
        self.tokenizer = RegexpTokenizer(r'\w+')

        def tokenize(x):
            if type(x) == str:
                return self.tokenizer.tokenize(x.lower())
            else:
                return []
            
        def encode(x):
            # Use 0 for unknown words (not in word_dict) by calling .get()
            x = tokenize(x)
            return [self.word_dict.get(word, 0) for word in x]
        
        # Cut or pad title and abstract
        def cut_or_pad_title(x):
            if len(x) > max_title_length:
                return x[:max_title_length]
            else:
                return x + [0]*(max_title_length - len(x)) 
        
        def cut_or_pad_abstract(x):
            if len(x) > max_abstract_length:
                return x[:max_abstract_length]
            else:
                return x + [0]*(max_abstract_length - len(x))
            
        # Encode user_id
        def encode_userid(x):
            return self.userid_dict.get(x, 0)
        
        # Encode user_id
        def encode_userid(x):
            return self.userid_dict.get(x, 0)
        
        # Encode history
        def encode_history(x):
            if type(x) != str:
                return []
            x = x.split(" ")
            return [self.news_dict.get(news_id, 0) for news_id in x]

        # get impressions and labels
        def get_impressions(x):
            x = x.split(" ")
            x = [imp.split("-")[0] for imp in x]
            return [self.news_dict.get(news_id, 0) for news_id in x]

        def get_labels(x):
            x = x.split(" ")
            x = [imp.split("-")[1] for imp in x]
            return [int(label) for label in x]
        
        # Cut or pad history
        def cut_or_pad(x):
            if len(x) > self.max_length:
                return x[-self.max_length:]
            else:
                return x + [0]*(self.max_length - len(x))

        # Load news data
        self.news_data = pd.read_csv(news_file, sep='\t', header=None)
        self.news_data.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_dict = {news_id: idx for idx, news_id in enumerate(self.news_data['news_id'])}

        # Encode title and abstract
        self.news_data['title_encode'] = self.news_data['title'].apply(encode)
        self.news_data['abstract_encode'] = self.news_data['abstract'].apply(encode)

        # Cut or pad title and abstract
        self.news_data['title_encode'] = self.news_data['title_encode'].apply(cut_or_pad_title)
        self.news_data['abstract_encode'] = self.news_data['abstract_encode'].apply(cut_or_pad_abstract)
        
        # Load user data
        self.user_data = pd.read_csv(user_file, sep='\t', header=None)
        self.user_data.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.user_data = self.user_data.dropna()

        # Encode user_id, history, impressions and get labels      
        self.user_data['user_id_encoded'] = self.user_data['user_id'].apply(encode_userid)
        self.user_data['history_encoded'] = self.user_data['history'].apply(encode_history)    
        self.user_data['impressions_encoded'] = self.user_data['impressions'].apply(get_impressions)
        self.user_data['labels'] = self.user_data['impressions'].apply(get_labels)

   
        # Save individual history lengths and impressions lengths
        self.user_data['history_length'] = self.user_data['history_encoded'].apply(len)
        self.user_data['impressions_length'] = self.user_data['impressions_encoded'].apply(len)

        # Get max lengths
        self.max_impressions_length = self.user_data['impressions_length'].max()
        self.max_history_length = self.user_data['history_length'].max()
        self.max_length = max(self.max_history_length, self.max_impressions_length)

        
        # Cut or pad history
        self.user_data['history_encoded'] = self.user_data['history_encoded'].apply(cut_or_pad)
        self.user_data['impressions_encoded'] = self.user_data['impressions_encoded'].apply(cut_or_pad)
        self.user_data['labels'] = self.user_data['labels'].apply(cut_or_pad)

        # Print max lengths
        print("Max history length: ", self.user_data['history_length'].max())
        print("Max impressions length: ", self.user_data['impressions_length'].max())


    def __len__(self):
        return len(self.user_data)


    def __getitem__(self, idx):
        # Get user_id
        user_id = self.user_data.iloc[idx]['user_id_encoded']

        # Get history
        history = self.user_data.iloc[idx]['history_encoded']
        history_length = self.user_data.iloc[idx]['history_length']

        # Get impressions and labels
        impressions = np.array(self.user_data.iloc[idx]['impressions_encoded'])
        impressions_length = self.user_data.iloc[idx]['impressions_length']
        labels = self.user_data.iloc[idx]['labels']

        # Get history as title and abstract
        history_title = [self.news_data.iloc[news_id]['title_encode'] for news_id in history]
        history_abstract = [self.news_data.iloc[news_id]['abstract_encode'] for news_id in history]
 
        # Get impressions as title as one long sequence
        impressions_title = [self.news_data.iloc[news_id]['title_encode'] for news_id in impressions]
        impressions_abstract = [self.news_data.iloc[news_id]['abstract_encode'] for news_id in impressions]

        

        # Convert to tensors
        user_id = th.tensor(user_id, dtype=th.long).to(self.device)
        history_title = th.tensor(history_title, dtype=th.long).to(self.device)        
        history_abstract = th.tensor(history_abstract, dtype=th.long).to(self.device)
        history_length = th.tensor(history_length, dtype=th.long).to(self.device)
        impressions_title = th.tensor(impressions_title, dtype=th.long).to(self.device)
        impressions_abstract = th.tensor(impressions_abstract, dtype=th.long).to(self.device)
        impressions_length = th.tensor(impressions_length, dtype=th.long).to(self.device)
        labels = th.tensor(labels, dtype=th.float).to(self.device)

        # Pad to max length
        history_title[history_length:,:] = 0
        history_abstract[history_length:,:] = 0

        # Pad to max length
        impressions_title[impressions_length:,:] = 0
        impressions_abstract[impressions_length:,:] = 0

        return user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels

#%%

# Define Data, Dataset and DataLoaders
train_behaviors_file = 'Data/MINDdemo_train/behaviors.tsv'
train_news_file = 'Data/MINDdemo_train/news.tsv'
word_dict_file = 'Data/MINDdemo_utils/word_dict_all.pkl'
user_dict_file = 'Data/MINDdemo_utils/uid2index.pkl'

valid_behaviors_file = 'Data/MINDdemo_dev/behaviors.tsv'
valid_news_file = 'Data/MINDdemo_dev/news.tsv'


import pickle

with open ("Data/MINDdemo_utils/word_dict.pkl", "rb") as f:
    word_dict = pickle.load(f)
with open ("Data/MINDdemo_utils/uid2index.pkl", "rb") as f:
    uid2index = pickle.load(f)

from dataclasses import dataclass

@dataclass
class HyperParams:
    batch_size: int
    title_size: int
    his_size: int
    wordDict_file: str
    userDict_file: str

hparamsdata = HyperParams(
    batch_size=32,
    title_size=20,
    his_size=50,
    wordDict_file=word_dict_file,
    userDict_file=user_dict_file,
)

TrainData = NewsDataset(train_behaviors_file, train_news_file, word_dict_file, userid_dict=uid2index)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=uid2index)

#%%

# Define DataLoaders
train_loader = DataLoader(TrainData, batch_size=hparamsdata.batch_size, shuffle=False, num_workers=0)


for batch in train_loader:
    user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels = batch
    break


# %%
