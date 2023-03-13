#%%
# Import packages

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer

Tokenizer = get_tokenizer('basic_english')
GloVe = torchtext.vocab.GloVe(name='840B', dim=300, cache='torchtext_data')

#%%
# Define the title encoder
class TitleEncoder(nn.Module):
    def __init__(self, word_dim=300, window_size=3, channel_size=300):
        super(TitleEncoder, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.Conv1d= nn.Conv1d(word_dim,channel_size,kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.v = nn.Parameter(th.rand(channel_size,1))
        self.vb = nn.Parameter(th.rand(1))
        self.Softmax = nn.Softmax(dim=0)

    def forward(self, W):

        W = self.dropout1(W)

        C = th.transpose(self.Conv1d(th.transpose(W,1,-1)),1,-1)
        C = self.dropout2(C)

        #*
        a = th.tanh( C @ self.v + self.vb) 
        alpha = self.Softmax(a)

        e = th.transpose(alpha,1,-1) @ C

        return e[:,0,:]


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
        title = self.TitleEncoder(W)


        return self.dropout(th.hstack([topic, title]))


# Define the user encoder
class UserEncoder(nn.Module):
    def __init__(self, user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300, device="cpu"):
        super(UserEncoder, self).__init__()
        self.seq_len = seq_len
        self.UserEmbedding = nn.Embedding(user_size, user_dim,padding_idx=0)
        self.NewsEncoder = NewsEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim)
        self.gru = nn.GRU(  input_size = word_dim+topic_dim+subtopic_dim, 
                            hidden_size = word_dim+topic_dim+subtopic_dim-user_dim, 
                            num_layers = 1,
                            # dropout = 0.2, 
                            batch_first=True)
        self.device = device
        self.dropout = nn.Dropout(0.2)

    def forward(self, users,topic,subtopic, W, src_len):
        b, n, t, _ = W.shape
        
        news_embed = th.zeros(b,n,500,device=self.device)
        for i in range(b):
            news_embed[i] = self.NewsEncoder(topic[i],subtopic[i], W[i])

        user_embed = self.UserEmbedding(users)
        # src_len_cpu = src_len.cpu()
        packed_news = nn.utils.rnn.pack_padded_sequence(news_embed, src_len.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        packed_outputs,hidden = self.gru(packed_news)




        user_s = self.dropout(hidden[-1])

        u = th.hstack([user_s, user_embed])         

        return u


# Define the LSTUR-ini model
class LSTUR_con(nn.Module):
    def __init__(self, user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300, device="cpu"):
        super(LSTUR_con, self).__init__()
        self.UserEncoder = UserEncoder(user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim, device)
        self.NewsEncoder = NewsEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim)
        self.device = device

    def forward(self, users,topic,subtopic, W, src_len, Candidate_topic,Candidate_subtopic,CandidateNews):
        b, n, t, _ = CandidateNews.shape
        Users = self.UserEncoder(users,topic,subtopic, W, src_len)

        Candidates =  th.zeros(b,n,500,device=self.device)
        for i in range(b):
            Candidates[i] = self.NewsEncoder(Candidate_topic[i],Candidate_subtopic[i], CandidateNews[i])
        

        Scores = th.zeros(b,n,device=self.device)
        for i in range(b):
            Scores[i] = Candidates[i] @ Users[i]

        return Scores


