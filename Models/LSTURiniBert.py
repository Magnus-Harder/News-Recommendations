#%%
# Import packages

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer

# Import libraries
import torch as th
import pandas as pd
import random
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, AdamW
from tqdm import tqdm

# Import BertTokenizer and BertModel
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Model = BertModel.from_pretrained("bert-base-uncased")
#Model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
#Model = BertForMaskedLM.from_pretrained('bert-base-uncased')

dim_bert = 768


#%%
# Define the title encoder
class TitleEncoder(nn.Module):
    def __init__(self, word_dim=300, window_size=3, channel_size=300):
        super(TitleEncoder, self).__init__()
        self.bert = Model

    def forward(self, W):

        O_bert = self.bert(W)
        return O_bert.pooler_output


# Define the topic encoder
class TopicEncoder(nn.Module):
    def __init__(self, topic_dim, subtopic_dim, topic_size, subtopic_size):
        super(TopicEncoder, self).__init__()
        self.topic_embed = nn.Embedding(topic_size, topic_dim, padding_idx=0)
        self.subtopic_embed = nn.Embedding(subtopic_size, subtopic_dim, padding_idx=0)
        

    def forward(self, topic, subtopic):
        
        topic = self.topic_embed(topic)
        subtopic = self.subtopic_embed(subtopic)


        return th.hstack([topic, subtopic])



# Define News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300):
        super(NewsEncoder, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.TitleEncoder = TitleEncoder(word_dim)
        self.TopicEncoder = TopicEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size)

    def forward(self, topic,subtopic, W):
        
        topic = self.TopicEncoder(topic, subtopic)

        with th.no_grad():
            title = self.TitleEncoder(W)


        return self.dropout(th.hstack([topic, title]))


# Define the user encoder
class UserEncoder(nn.Module):
    def __init__(self, user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300, device="cpu"):
        super(UserEncoder, self).__init__()
        self.seq_len = seq_len
        self.UserEmbedding = nn.Embedding(user_size, user_dim,padding_idx=0)
        self.NewsEncoder = NewsEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim)
        self.gru = nn.GRU(  input_size = dim_bert+topic_dim+subtopic_dim, 
                            hidden_size = dim_bert+topic_dim+subtopic_dim, 
                            num_layers = 1,
                            # dropout = 0.2, 
                            batch_first=True)
        self.device = device
        self.dropout = nn.Dropout(0.2)
        self.news_size = dim_bert+topic_dim+subtopic_dim

    def forward(self, users,topic,subtopic, W, src_len):
        b, n, t = W.shape
        
        news_embed = th.zeros(b,n,self.news_size,device=self.device)
        for i in range(b):
            news_embed[i] = self.NewsEncoder(topic[i],subtopic[i], W[i])

        user_embed = self.UserEmbedding(users)
        # src_len_cpu = src_len.cpu()
        packed_news = nn.utils.rnn.pack_padded_sequence(news_embed, src_len.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        packed_outputs,hidden = self.gru(packed_news, user_embed.unsqueeze(0))

        user_s = self.dropout(hidden[-1])
         
        return user_s


# Define the LSTUR-ini model
class LSTURbert(nn.Module):
    def __init__(self, user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300, device="cpu"):
        super(LSTURbert, self).__init__()
        self.UserEncoder = UserEncoder(user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim, device)
        self.NewsEncoder = NewsEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim)
        self.device = device
        self.news_size = dim_bert+topic_dim+subtopic_dim

    def forward(self, users,topic,subtopic, W, src_len, Candidate_topic,Candidate_subtopic,CandidateNews):
        b, n, t = CandidateNews.shape
        Users = self.UserEncoder(users,topic,subtopic, W, src_len)

        Candidates =  th.zeros(b,n,self.news_size,device=self.device)
        for i in range(b):
            Candidates[i] = self.NewsEncoder(Candidate_topic[i],Candidate_subtopic[i], CandidateNews[i])
        

        Scores = th.zeros(b,n,device=self.device)
        for i in range(b):
            Scores[i] = Candidates[i] @ Users[i]

        return Scores


