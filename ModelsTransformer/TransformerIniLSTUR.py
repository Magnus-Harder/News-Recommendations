#%%
import torch as th
from torch import nn
from ModelsTransformer.NewsEncoder import NewsEncoder

#%%
class PositionalEncoding(nn.Module):
    def __init__(self, T, d_model):
        super().__init__()

        # Define needed built in functions and parameters
        self.d_model = d_model
        self.T = T
        self.PositionalEncoding = nn.Parameter(th.zeros((int(T), int(d_model))), requires_grad=False)
        self.init_positional_encoding()

    # Initialize positional encoding matrix
    def init_positional_encoding(self):
        position = th.arange(0, self.T, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, self.d_model, 2,dtype=th.float) * (-th.log(th.tensor([10000.0])) / self.d_model))
        self.PositionalEncoding[:, 0::2] = th.sin(position * div_term)
        self.PositionalEncoding[:, 1::2] = th.cos(position * div_term)

    # Add positional encoding to input
    def forward(self, X):
        return X + self.PositionalEncoding


class UserEncoder(nn.Module):
    def __init__(self, attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit,user_size, word_vectors, NewsEncoder, device="cpu"):
        super(UserEncoder, self).__init__()

        # User embedding
        self.UserEmbedding = nn.Embedding(user_size, gru_unit,padding_idx=0)
        
        # News Encoder
        self.NewsEncoder = NewsEncoder
        #self.NewsEncoder = NewsEncoder(attention_dim, word_emb_dim, dropout, filter_num, windows_size, gru_unit, word_vectors, device)
        
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


    def forward(self, user_id, history_title,history_length):
        b, n, t, = history_title.shape
        
        news_embed = th.zeros(b,n,self.news_size,device=self.device)
        for i in range(b):
            news_embed[i] = self.NewsEncoder(history_title[i])

        user_embed = self.UserEmbedding(user_id)
        
        # Pack the news embedding
        packed_news = nn.utils.rnn.pack_padded_sequence(news_embed, history_length.cpu(), batch_first=True, enforce_sorted=False).to(self.device)
        output,hidden = self.gru(packed_news, user_embed.unsqueeze(0))

        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Batch, User
        user_s = hidden.squeeze(0)
        
        return user_s, output




class lstransformer(nn.Module):
        def __init__(self, his_size, d_model, ffdim, nhead, num_layers, user_vocab_size, attention_dim, word_emb_dim, filter_num, window_size, word_vectors ,device,dropout=0.2):
            super().__init__()
            self.num_layers = num_layers

            self.his_size = his_size
            # RNN Encoder
            self.newsencoder = NewsEncoder(attention_dim, word_emb_dim, dropout, filter_num, window_size, word_vectors, device)
            self.userencoder = UserEncoder(attention_dim, word_emb_dim, dropout, filter_num, window_size, d_model,user_vocab_size, word_vectors, self.newsencoder, device)
            
            # Decoder
            self.decoderlayer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer = self.decoderlayer, num_layers = self.num_layers)
            

            #Newsencoder and userembedder
            
            #self.UserEmbedding = nn.Embedding(user_vocab_size, d_model ,padding_idx=0)

            self.device = device

            #Final linear layer
            self.outlayer = nn.Linear(d_model, 1)
            self.softmax = nn.Softmax(1)

            # Dropout_layers
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        def forward(self, user_id, embed_his, his_key_mask, candidates):

            # Get history length
            his_length = self.his_size - th.sum(his_key_mask,dim=1)
            print(his_length)

            user_rep, memory = self.userencoder(user_id, embed_his, his_length)

            # dropouts
            memory = self.dropout2(memory)

            # Embed candidates
            embed_cand = th.empty((candidates.shape[0],candidates.shape[1],400),device=self.device)
            for i in range(candidates.shape[0]):
                embed_cand[i] = self.newsencoder(candidates[i])

            #Dropouts
            embed_cand = self.dropout3(embed_cand)

            # Add user embedding to front of candidates
            User_embed_cant = th.cat((user_rep.unsqueeze(1),embed_cand),dim=1)

            #Decode candidates with memory
            decoded = self.decoder(User_embed_cant,memory)

            #Dropouts
            decoded = self.dropout4(decoded)
            
            #Final layer
            out = self.outlayer(decoded)

            return out[:,1:]









# %%
