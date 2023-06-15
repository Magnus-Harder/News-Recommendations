#%%
# Import pytorch elements
import torch as th
from torch.utils.data import Dataset
from torchtext import vocab

# Import basic packages
import pandas as pd
import numpy as np


class NewsDataset(Dataset):
    def __init__(self, user_file, news_file, News_dict, max_history_length=50, Category_vocab = None, Subcategory_vocab = None, userid_dict = None,train=True, npratio=4, mask_prob= 0.5,device='cpu',transformer=False):
        
        self.device = device
        self.mask_prob = mask_prob
        self.train = train
        self.News_dict = News_dict
        self.max_history_length = max_history_length
        self.npratio = npratio
        self.transformer = transformer



        # Load news data
        self.news_data = pd.read_csv(news_file, sep='\t', header=None)
        self.news_data.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

        # Load user data
        self.user_data = pd.read_csv(user_file, sep='\t', header=None)
        self.user_data.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.user_data = self.user_data.dropna()
        

        # Get category_dict
        if Category_vocab is None:
            self.Category_vocab = vocab.build_vocab_from_iterator([[Category] for Category in self.news_data['category']], specials=['<unk>'])
            self.Category_vocab.set_default_index(self.Category_vocab['<unk>'])
        else:
            self.Category_vocab = Category_vocab

        # Get subcategory_dict
        if Subcategory_vocab is None:
            self.Subcategory_vocab = vocab.build_vocab_from_iterator([[Category] for Category in self.news_data['subcategory']], specials=['<unk>'])
            self.Subcategory_vocab.set_default_index(self.Subcategory_vocab['<unk>'])
        else:
            self.Subcategory_vocab = Subcategory_vocab

        # Define Category and Subcategory news_dict

        self.News_subcategory_dict = {}
        self.News_category_dict = {}

        for news_id, category, subcategory in zip(self.news_data['news_id'], self.news_data['category'], self.news_data['subcategory']):
            encoded_id = self.News_dict.get(news_id, 0)

            self.News_subcategory_dict[encoded_id] = self.Subcategory_vocab[subcategory]
            self.News_category_dict[encoded_id] = self.Category_vocab[category]


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
            return [self.News_dict[news_id] for news_id in x]

        self.user_data['history_encoded'] = self.user_data['history'].apply(encode_history)

        # get impressions and labels
        def get_impressions(x):
            x = x.split(" ")
            x = [imp.split("-")[0] for imp in x]
            return [self.News_dict[news_id] for news_id in x]

        def get_labels(x):
            x = x.split(" ")
            x = [imp.split("-")[1] for imp in x]
            return [int(label) for label in x]

        self.user_data['impressions_encoded'] = self.user_data['impressions'].apply(get_impressions)
        self.user_data['labels'] = self.user_data['impressions'].apply(get_labels)
        
        # Save number of positive impressions
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
                        negative_impressions.append(self.News_dict[news_id])
                
                return negative_impressions
            
            self.user_data['negative_impressions'] = self.user_data['impressions'].apply(get_negative_impressions)
            
            # get positive impressions
            def get_positive_impressions(impressions):
                impressions = impressions.split(" ")
                
                positive_impressions = []

                for impression in impressions:
                    news_id, label = impression.split("-")
                    if label == "1":
                        positive_impressions.append(self.News_dict[news_id])
                
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

        # Get history as category and subcategory
        history_subcategory = [self.News_subcategory_dict.get(news_id,0) for news_id in history]
        history_category = [self.News_category_dict.get(news_id,0) for news_id in history]

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
            
            # Get impressions as category and subcategory
            impressions_subcategory = [self.News_subcategory_dict.get(news_id,0) for news_id in impressions]
            impressions_category = [self.News_category_dict.get(news_id,0) for news_id in impressions]

            # Get labels
            labels = np.array([1]+[0]*self.npratio)

            # Get impressions length
            impressions_length = len(impressions)

            # Shuffle impressions if Transformer
            if self.transformer:
                shuffle_idx = np.arange(impressions_length)
                np.random.shuffle(shuffle_idx)
                impressions = impressions[shuffle_idx]
                impressions_category = [impressions_category[i] for i in shuffle_idx]
                impressions_subcategory = [impressions_subcategory[i] for i in shuffle_idx]
                labels = labels[shuffle_idx]


        else:
            # Get impressions as category and subcategory
            impressions_subcategory = [self.News_subcategory_dict.get(news_id,0) for news_id in impressions]
            impressions_category = [self.News_category_dict.get(news_id,0) for news_id in impressions]
            impressions_length = self.user_data.iloc[idx]['impressions_length']



        # Convert to tensors
        user_id = th.tensor(user_id, dtype=th.long).to(self.device)
        history = th.tensor(history, dtype=th.long).to(self.device)
        history_category = th.tensor(history_category, dtype=th.long).to(self.device)
        history_subcategory = th.tensor(history_subcategory, dtype=th.long).to(self.device)
        history_length = th.tensor(history_length, dtype=th.long).to(self.device)


        # Impression tensors
        impressions = th.tensor(impressions, dtype=th.long).to(self.device)
        impressions_category = th.tensor(impressions_category, dtype=th.long).to(self.device)
        impressions_subcategory = th.tensor(impressions_subcategory, dtype=th.long).to(self.device)
        impressions_length = th.tensor(impressions_length, dtype=th.long).to(self.device)
        labels = th.tensor(labels, dtype=th.float).to(self.device)


        return user_id, history, history_category, history_subcategory,  history_length, impressions, impressions_category, impressions_subcategory, impressions_length, labels


# %%
