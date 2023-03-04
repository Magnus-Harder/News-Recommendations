#%%
# Import packages

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer

Tokenizer = get_tokenizer('basic_english')
GloVe = torchtext.vocab.GloVe(name='840B', dim=300, cache='torchtext_data')


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
        self.TitleEncoder = TitleEncoder(word_dim)
        self.TopicEncoder = TopicEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size)

    def forward(self, topic,subtopic, W):
        
        topic = self.TopicEncoder(topic, subtopic)
        title = self.TitleEncoder(W)


        return th.hstack([topic, title])


# Define the user encoder
class UserEncoder(nn.Module):
    def __init__(self, user_dim, user_size,seq_len,topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim=300, device="cpu"):
        super(UserEncoder, self).__init__()
        self.seq_len = seq_len
        self.UserEmbedding = nn.Embedding(user_size, user_dim,padding_idx=0)
        self.NewsEncoder = NewsEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size, word_dim)
        self.gru = nn.GRU(word_dim+topic_dim+subtopic_dim, word_dim+topic_dim+subtopic_dim-user_dim, seq_len, batch_first=True)
        self.device = device

    def forward(self, users,topic,subtopic, W, src_len):
        b, n, t, _ = W.shape
        
        news_embed = th.zeros(b,n,500,device=self.device)
        for i in range(b):
            news_embed[i] = self.NewsEncoder(topic[i],subtopic[i], W[i])

        user_embed = self.UserEmbedding(users)
        # src_len_cpu = src_len.cpu()
        packed_news = nn.utils.rnn.pack_padded_sequence(news_embed, src_len.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        packed_outputs,hidden = self.gru(packed_news)




        user_s = hidden[-1]

        u = th.hstack([user_embed, user_s])         

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




#%%
# device= "cuda"
# Users = th.tensor([1,2,3,4,5,6,7,8,9,0]).to(device)
# topics =  th.randint(0,10,(10,10)).to(device)
# subtopics = th.randint(0,10,(10,10)).to(device)
# W_batch = th.rand(10,10,10,300).to(device)
# Candidate_topic = th.randint(0,10,(10,5)).to(device)
# Candidate_subtopic = th.randint(0,10,(10,5)).to(device)
# CandidateNews = th.rand(10,5,10,300).to(device)





# UserEncoder_module = UserEncoder(
#     user_dim=300,
#     user_size=10,
#     seq_len=10,
#     topic_dim=100,
#     subtopic_dim=100,
#     topic_size=10,
#     subtopic_size=10,
#     word_dim=300
# )

# LSTUR_con_module = LSTUR_con(
#     user_dim=300,
#     user_size=10,
#     seq_len=10,
#     topic_dim=100,
#     subtopic_dim=100,
#     topic_size=10,
#     subtopic_size=10,
#     word_dim=300,
#     device=device
# ).to(device)

# src_len = th.tensor([10,5,6,8,10,9,6,8,9,3]).to(device)
# target = th.tensor([1,0,4,0,2,0,3,0,1,2]).to(device)


# optimizer = th.optim.Adam(LSTUR_con_module.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()

# for i in range(15):
#     optimizer.zero_grad()
#     Scores = LSTUR_con_module(Users,topics,subtopics,W_batch,src_len,Candidate_topic,Candidate_subtopic,CandidateNews)

#     loss = loss_fn(Scores, target)
#     loss.backward()
#     optimizer.step()

#     print(f"Predicted {Scores.argmax(dim=1)} with loss {loss.item()}")




# %%
