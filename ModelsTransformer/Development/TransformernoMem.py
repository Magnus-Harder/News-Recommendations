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





class lstransformer(nn.Module):
        def __init__(self, his_size, d_model, ffdim, nhead, num_layers, user_vocab_size, attention_dim, word_emb_dim, filter_num, window_size, word_vectors ,device,dropout=0.2):
            super().__init__()
            self.num_layers = num_layers

            # Positional encoding
            self.positional_encoding = PositionalEncoding(his_size,d_model)

            # Encoder 
            self.encoderlayer = nn.TransformerEncoderLayer(d_model,nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer = self.encoderlayer, num_layers = self.num_layers)
            
            # Decoder
            self.decoderlayer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer = self.decoderlayer, num_layers = self.num_layers)
            

            #Newsencoder and userembedder
            self.newsencoder = NewsEncoder(attention_dim, word_emb_dim, dropout, filter_num, window_size, word_vectors, device)
            self.UserEmbedding = nn.Embedding(user_vocab_size, d_model ,padding_idx=0)

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

            # Encode history
            encoded_his = th.empty((embed_his.shape[0],embed_his.shape[1],400),device=self.device)

            for i in range(embed_his.shape[0]):
                 encoded_his[i] = self.newsencoder(embed_his[i])
            
            
            # Dropouts
            encoded_his = self.dropout1(encoded_his)

            # Add positional encoding to history
            encoded_his = self.positional_encoding(encoded_his)

            #embed_his = self.newsencoder(embed_his)
            memory = self.encoder(encoded_his, src_key_padding_mask = his_key_mask)
            
            # Zero the memory
            memory = memory * 0
            
            # dropouts
            memory = self.dropout2(memory)

            embed_cand = th.empty((candidates.shape[0],candidates.shape[1],400),device=self.device)
            for i in range(candidates.shape[0]):
                embed_cand[i] = self.newsencoder(candidates[i])

            # Embed user id
            users = self.UserEmbedding(user_id)

            #Dropouts
            embed_cand = self.dropout3(embed_cand)

            # Add user embedding to front of candidates
            User_embed_cant = th.cat((users.unsqueeze(1),embed_cand),dim=1)

            #Decode candidates with memory
            decoded = self.decoder(User_embed_cant,memory)

            #Dropouts
            decoded = self.dropout4(decoded)
            
            #Final layer
            out = self.outlayer(decoded)

            return out[:,1:]









# %%
