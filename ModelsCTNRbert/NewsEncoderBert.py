# Import packages
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


#Define the topic encoder
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
    def __init__(self, topic_dim,subtopic_dim,topic_size,subtopic_size, dropout, title_vectors, abstract_vectors, device="cpu"):
        super(NewsEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.TitleEncoder = nn.Embedding.from_pretrained(title_vectors, freeze=True)
        self.abstractEncoder = nn.Embedding.from_pretrained(abstract_vectors, freeze=True)
        self.TopicEncoder = TopicEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size)

        self.dim_news = title_vectors.shape[1] + topic_dim + subtopic_dim

        # attention for title and abstract
        self.attention_weight = nn.Parameter(th.rand(2))

        # Linear layer
        self.linear = nn.Linear(self.dim_news, self.dim_news)


    def forward(self, news, news_topic, news_subtopic):
        
        title = self.TitleEncoder(news)
        abstract = self.abstractEncoder(news)

        # Combine title and abstract
        self.alpha = F.softmax(self.attention_weight, dim=0)
        news = self.alpha[0] * title + self.alpha[1] * abstract

        topic = self.TopicEncoder(news_topic, news_subtopic)
        
        # Concatenate title and topic
        article = th.cat((news, topic), dim=1)

        # Apply linear layer
        article = self.linear(article)

        return article

        