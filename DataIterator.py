#%%
# Import pytorch elements
import torch as th
from torch.utils.data import Dataset, DataLoader


# Import text utils
from nltk.tokenize import RegexpTokenizer
import torchtext

# Import basic packages
import pandas as pd
import numpy as np
import pickle

# import GloVe

# import word_dict
with open('Data/MINDdemo_utils/word_dict_all.pkl', 'rb') as f:
    word_dict = pickle.load(f)


class NewsDataset(Dataset):
    def __init__(self, user_file, news_file, word_dict_file, max_history_length=50, max_title_length=30, max_abstract_length=100,userid_dict = None,train=True, npratio=4, mask_prob= 0.5,device='cpu',transformer=False):
        
        self.device = device
        self.mask_prob = mask_prob
        self.train = train
        self.max_history_length = max_history_length
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.npratio = npratio
        self.transformer = transformer
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
 

        # Load news data
        self.news_data = pd.read_csv(news_file, sep='\t', header=None)
        self.news_data.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        self.news_dict = {news_id: idx for idx, news_id in enumerate(self.news_data['news_id'])}

        # Encode title and abstract
        self.news_data['title_encode'] = self.news_data['title'].apply(encode)
        self.news_data['abstract_encode'] = self.news_data['abstract'].apply(encode)

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

        self.news_data['title_encode'] = self.news_data['title_encode'].apply(cut_or_pad_title)
        self.news_data['abstract_encode'] = self.news_data['abstract_encode'].apply(cut_or_pad_abstract)

        
        # Load user data
        self.user_data = pd.read_csv(user_file, sep='\t', header=None)
        self.user_data.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.user_data = self.user_data.dropna()

        # Get user_id_dict
        if userid_dict is None:
            self.userid_dict = {user_id: idx+1 for idx, user_id in enumerate(self.user_data['user_id'].unique())}
        else:
            self.userid_dict = userid_dict

        # Encode user_id
        def encode_userid(x):
            return self.userid_dict.get(x, 0)
        
        self.user_data['user_id_encoded'] = self.user_data['user_id'].apply(encode_userid)

        # Encode history
        def encode_history(x):
            if type(x) != str:
                return []
            x = x.split(" ")
            return [self.news_dict.get(news_id, 0) for news_id in x]

        self.user_data['history_encoded'] = self.user_data['history'].apply(encode_history)

        # get impressions and labels
        def get_impressions(x):
            x = x.split(" ")
            x = [imp.split("-")[0] for imp in x]
            return [self.news_dict.get(news_id, 0) for news_id in x]

        def get_labels(x):
            x = x.split(" ")
            x = [imp.split("-")[1] for imp in x]
            return [int(label) for label in x]
        
        self.user_data['impressions_encoded'] = self.user_data['impressions'].apply(get_impressions)
        self.user_data['labels'] = self.user_data['impressions'].apply(get_labels)

        self.user_data['n_postive'] = self.user_data['labels'].apply(lambda x: sum(x))
        self.max_positive = self.user_data['n_postive'].max()

        # Save individual history lengths and impressions lengths
        self.user_data['history_length'] = self.user_data['history_encoded'].apply(len)
        self.user_data['impressions_length'] = self.user_data['impressions_encoded'].apply(len)

        self.max_impressions_length = self.user_data['impressions_length'].max()


        # Cut or pad history
        def cut_or_pad(x):
            if len(x) > max_history_length:
                return x[-max_history_length:]
            else:
                return x + [0]*(max_history_length - len(x))

        self.user_data['history_encoded'] = self.user_data['history_encoded'].apply(cut_or_pad)

        # Save new history lengths
        self.user_data['history_length'][self.user_data['history_length'] > 50] = 50

        
        if train:
             # get negative impressions
            def get_negative_impressions(impressions):

                impressions = impressions.split(" ")
                
                negative_impressions = []


                for impression in impressions:
                    news_id, label = impression.split("-")

                    if label == "0":
                        negative_impressions.append(self.news_dict[news_id])
                
                return negative_impressions
            
            self.user_data['negative_impressions'] = self.user_data['impressions'].apply(get_negative_impressions)
            
            # get positive impressions
            def get_positive_impressions(impressions):
                impressions = impressions.split(" ")
                
                positive_impressions = []

                for impression in impressions:
                    news_id, label = impression.split("-")
                    if label == "1":
                        positive_impressions.append(self.news_dict[news_id])
                
                return positive_impressions
            
            self.user_data['positive_impressions'] = self.user_data['impressions'].apply(get_positive_impressions)

            # Explode user data for each impression if training
            self.user_data = self.user_data.explode('positive_impressions')
            self.user_data.index = np.arange(len(self.user_data))

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
        labels = self.user_data.iloc[idx]['labels']

        # Get history as title and abstract
        history_title = [self.news_data.iloc[news_id]['title_encode'] for news_id in history]
        history_abstract = [self.news_data.iloc[news_id]['abstract_encode'] for news_id in history]

        # Get n_positive
        n_positive = self.user_data.iloc[idx]['n_postive']

        # Sample negative impressions 
        if self.train:
            positive_sample = self.user_data.iloc[idx]['positive_impressions']
            negative_sample = self.user_data.iloc[idx]['negative_impressions']
            if len(negative_sample) >= self.npratio:
                negative_sample = np.random.choice(negative_sample, size=self.npratio, replace=False)
            else:
                negative_sample = np.random.choice(negative_sample, size=self.npratio, replace=True)
            # Mask user
            if np.random.rand() < self.mask_prob:
                user_id = 0
            
            impressions = np.array([positive_sample, *negative_sample])
            
            # Get impressions as title and abstract
            impressions_title = [self.news_data.iloc[news_id]['title_encode'] for news_id in impressions]
            impressions_abstract = [self.news_data.iloc[news_id]['abstract_encode'] for news_id in impressions]

            # Get labels
            labels = np.array([1]+[0]*self.npratio)

            # Get impressions length
            impressions_length = len(impressions)

            # Shuffle impressions if Transformer
            if self.transformer:
                shuffle_idx = np.arange(impressions_length)
                np.random.shuffle(shuffle_idx)
                impressions = impressions[shuffle_idx]
                impressions_title = [impressions_title[i] for i in shuffle_idx]
                impressions_abstract = [impressions_abstract[i] for i in shuffle_idx]
                labels = labels[shuffle_idx]


        else:
            # Get impressions as title as one long sequence
            impressions_title = [self.news_data.iloc[news_id]['title_encode'] for news_id in impressions]
            impressions_abstract = [self.news_data.iloc[news_id]['abstract_encode'] for news_id in impressions]
        
            impressions_length = self.user_data.iloc[idx]['impressions_length']



        # Convert to tensors
        user_id = th.tensor(user_id, dtype=th.long).to(self.device)
        history_title = th.tensor(history_title, dtype=th.long).to(self.device)        
        history_abstract = th.tensor(history_abstract, dtype=th.long).to(self.device)
        history_length = th.tensor(history_length, dtype=th.long).to(self.device)
        impressions_title = th.tensor(impressions_title, dtype=th.long).to(self.device)
        impressions_abstract = th.tensor(impressions_abstract, dtype=th.long).to(self.device)
        impressions_length = th.tensor(impressions_length, dtype=th.long).to(self.device)
        labels = th.tensor(labels, dtype=th.float).to(self.device)

        history_title[history_length:,:] = 0
        history_abstract[history_length:,:] = 0


        return user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive

