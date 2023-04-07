#%%
# Import packages
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# Define the title encoder
class TitleEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit , word_vectors, device="cpu"):
        super(TitleEncoder, self).__init__()

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Convoilutional Layer
        self.Conv1d= nn.Conv1d(word_emb_dim, filter_num, kernel_size=windows_size, stride=1, padding=1)
        
        # Attention Layer
        self.v = nn.Parameter(th.rand(filter_num,attention_dim))
        self.vb = nn.Parameter(th.rand(attention_dim))
        self.q = nn.Parameter(th.rand(attention_dim,1))
        self.Softmax = nn.Softmax(dim=0)
        
        # Define device
        self.device = device

        # Define word embedding
        self.word_embedding = nn.Embedding.from_pretrained(th.tensor(word_vectors,dtype=th.float32), freeze=False, padding_idx=0)

        # Initialize the weights as double
        nn.init.xavier_uniform_(self.Conv1d.weight)
        nn.init.zeros_(self.Conv1d.bias)
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.vb)
        nn.init.xavier_uniform_(self.q)

        

    def forward(self, encoded_title):

        # Convert the encoded title to word embedding
        W = self.word_embedding(encoded_title)
        W = self.dropout1(W)        

        # Convolutional Layer
        C = th.transpose(self.Conv1d(th.transpose(W,1,-1)),1,-1)
        C = F.relu(C)
        C = self.dropout2(C)
        
        # Attention Layer
        seq_len, n_word, channel_size = C.shape

        a = th.tanh( C @ self.v + self.vb)
        a = a @ self.q
        a = a.squeeze(-1)

        alpha = self.Softmax(a)
        
        # shape e: seq_len, channel_size
        e = th.zeros(seq_len, channel_size, device=self.device)

        for i in range(seq_len):
            e[i] = alpha[i] @ C[i]


        return e


#Define the topic encoder
# class TopicEncoder(nn.Module):
#     def __init__(self, topic_dim, subtopic_dim, topic_size, subtopic_size):
#         super(TopicEncoder, self).__init__()
#         self.topic_embed = nn.Embedding(topic_size, topic_dim, padding_idx=0)
#         self.subtopic_embed = nn.Embedding(subtopic_size, subtopic_dim, padding_idx=0)
        

#     def forward(self, topic, subtopic):
        
#         topic = self.topic_embed(topic)
#         subtopic = self.subtopic_embed(subtopic)


#         return th.hstack([topic, subtopic])



# Define News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, word_vectors, device="cpu"):
        super(NewsEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.TitleEncoder = TitleEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, word_vectors, device)
        #self.TopicEncoder = TopicEncoder(topic_dim, subtopic_dim, topic_size, subtopic_size)

        # Initialize the weights


    def forward(self, news_titles):
        
        title = self.TitleEncoder(news_titles)

        out  = self.dropout(title)

        return out


# Define the user encoder
class UserEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit,user_size, word_vectors, device="cpu"):
        super(UserEncoder, self).__init__()

        # User embedding
        self.UserEmbedding = nn.Embedding(user_size, gru_unit,padding_idx=0)
        
        # News Encoder
        self.NewsEncoder = NewsEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, word_vectors, device)
        
        # GRU
        self.gru = nn.GRU(  input_size = filter_num, 
                            hidden_size = gru_unit, 
                            num_layers = 1,
                            batch_first=True)
        
        # Dropout and device
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # Define parameter intialization
        nn.init.zeros_(self.UserEmbedding.weight)
        #nn.init.xavier_uniform_(self.gru.weight)

        # Save news embedding size
        self.news_size = filter_num


    def forward(self, user_id, history_title, history_length):
        b, n, t, = history_title.shape
        
        news_embed = th.zeros(b,n,self.news_size,device=self.device)
        for i in range(b):
            news_embed[i] = self.NewsEncoder(history_title[i])

        user_embed = self.UserEmbedding(user_id)
        
        
        packed_news = nn.utils.rnn.pack_padded_sequence(news_embed, history_length.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        packed_outputs,hidden = self.gru(packed_news, user_embed.unsqueeze(0))

        # Batch, User
        user_s = hidden.squeeze(0)
         
        return user_s


# Define the LSTUR-ini model
class LSTURini(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, user_size, word_vectors ,device="cpu"):
        super(LSTURini, self).__init__()

        # User Encoder
        self.UserEncoder = UserEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, user_size, word_vectors, device)
        
        # News Encoder
        self.NewsEncoder = NewsEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, word_vectors, device)
        
        # Device
        self.device = device

        # Save news embedding size
        self.news_size = filter_num

    def forward(self, user_id, history_title, history_length, impressions_title, n_positive):
        
        # User embedding
        Users = self.UserEncoder(user_id, history_title, history_length)

        if self.training:
            b, p , n, t= impressions_title.shape
            Scores = th.zeros(b,p,n,device=self.device)

            for i in range(b):
                for j in range(n_positive[i]):
                    Scores[i,j] = self.NewsEncoder(impressions_title[i,j]) @ Users[i]

        else:
            b, n, t = impressions_title.shape
            Candidates =  th.zeros(b,n,self.news_size,device=self.device)
            for i in range(b):
                Candidates[i] = self.NewsEncoder(impressions_title[i])
            

            Scores = th.zeros(b,n,device=self.device)
            for i in range(b):
                Scores[i] = Candidates[i] @ Users[i]

        return Scores


# %%
#
