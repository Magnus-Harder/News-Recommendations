#%%
# Import packages

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext

GloVe = torchtext.vocab.GloVe(name='840B', dim=300, cache='torchtext_data')


# Define the title encoder
class TitleEncoder(nn.Module):
    def __init__(self, word_dim=300, window_size=3, channel_size=300):
        super(TitleEncoder, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.Conv1 = nn.Conv1d(word_dim,channel_size,kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.v = nn.Parameter(th.rand(channel_size, 1))
        self.vb = nn.Parameter(th.rand(1))
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, W):

        W = self.dropout1(W)

        C = self.Conv1(W.T).T
        C = self.dropout2(C)

        #*
        a = th.tanh( C @ self.v + self.vb) 
        alpha = self.Softmax(a)

        e = alpha.T @ C

        return e


# Define the topic encoder
class TopicEncoder(nn.Module):
    def __init__(self, topic_dim, subtopic_dim, topic_size, subtopic_size):
        super(TopicEncoder, self).__init__()
        self.title_embed = nn.Embedding(topic_size, topic_dim)
        self.subtopic_embed = nn.Embedding(subtopic_size, subtopic_dim)
        

    def forward(self, topic, subtopic):
        
        topic = self.title_embed(topic)
        subtopic = self.subtopic_embed(subtopic)
        
        return th.hstack([topic, subtopic])



# Define News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, topic_dim, subtopic_dim, topic_size, subtopic_size, word_size, word_dim=200):
        super(NewsEncoder, self).__init__()
        self.TitleEncoder = TitleEncoder()
        self.TopicEncoder = TopicEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size)

    def forward(self, topic,subtopic, W):
        
        topic = self.TopicEncoder(topic, subtopic)
        title = self.TitleEncoder(W)


        return th.hstack([topic, title])
