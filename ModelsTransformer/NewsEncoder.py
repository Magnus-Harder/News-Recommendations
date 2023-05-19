# Import packages
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TitleEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, word_vectors, device="cpu"):
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

        mask = (encoded_title != 0).float()
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

        alpha = alpha * mask
        
        # shape e: seq_len, channel_size
        e = th.zeros(seq_len, channel_size, device=self.device)

        for i in range(seq_len):
            e[i] = alpha[i] @ C[i]

        return e



# Define News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, word_vectors, device="cpu"):
        super(NewsEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.TitleEncoder = TitleEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, word_vectors, device)


    def forward(self, news_titles):
        
        title = self.TitleEncoder(news_titles)

        out  = self.dropout(title)

        return out