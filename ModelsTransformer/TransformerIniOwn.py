#%%
import torch as th
from torch import nn
from ModelsTransformer.NewsEncoder import NewsEncoder
import math

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

# Define Attention class
class Attention(nn.Module):
    def __init__(self, dk,dropout=0.1):
        super().__init__()

        # Define needed built in functions
        self.Softmax = nn.Softmax(dim=1)
        self.sqrt_dk = math.sqrt(dk)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,Q,K,V,mask=None):
    
        # Query key dot product
        QK = Q @ th.transpose(K,1,2) / self.sqrt_dk
        QK = self.dropout(QK)

        # Apply mask if not None
        if mask is not None:
            QK = QK + mask
            print(QK.shape)
            print(mask.shape)

        # Get attention weights
        A = th.transpose(self.Softmax(th.transpose(QK,1,2)),1,2) @ V

        return A


# Define multi head attention
class MHA(nn.Module):
    def __init__(self,T, d_model,dk=256,dv=512, nhead = 8,dropout=0.1):
        super().__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.nhead = nhead
        self.dk = dk
        self.dv = dv
        self.T = T
        
        # Define Q,K,V 
        self.Qs = nn.Linear(d_model,int(dk*nhead))
        self.Ks = nn.Linear(d_model,int(dk*nhead))
        self.Vs = nn.Linear(d_model,int(dv*nhead))

        # Define attention layer
        self.Attention = Attention(dk,dropout)

        # Define output layer    
        self.out = nn.Linear(int(dv*nhead),d_model)

    def forward(self,Q,K,V,mask=None):

        # Intialize Q,K,V
        Qs = self.Qs(Q)
        Ks = self.Ks(K)
        Vs = self.Vs(V)

        # Reshape Q,K,V to batch heads
        Qs = th.transpose(Qs.reshape(self.nhead,self.dk,self.T),1,2)
        Ks = th.transpose(Ks.reshape(self.nhead,self.dk,self.T),1,2)
        Vs = th.transpose(Vs.reshape(self.nhead,self.dv,self.T),1,2)

        # Get each attention heads
        A = self.Attention(Qs,Ks,Vs,mask)
        A = th.transpose(A,1,-1).reshape(-1,self.T).T        
        
        # Apply linear layer to get input dimensions
        return self.out(A)


# Define Feedforward class
class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.Linear1 = nn.Linear(d_model,d_ff)
        self.Linear2 = nn.Linear(d_ff,d_model)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,X):
        return self.Linear2(self.ReLU(self.Linear1(X)))

# Define LayerNorm class
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        #self.d_model = d_model
        self.LayerNorm = nn.LayerNorm(d_model)
    def forward(self,X):
        return self.LayerNorm(X)


# Define Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self,T, d_model, nhead, d_ff, dk, dv, dropout=0.1):
        super().__init__()

        # Define needed layers
        self.MHA = MHA(T, d_model, dk, dv, nhead)
        self.LayerNorm1 = LayerNorm(d_model)
        self.Feedforward = Feedforward(d_model, d_ff)
        self.LayerNorm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,X,mask=None):
        
        # Get multi head attention
        mha = self.MHA(X,X,X,mask)

        # Add residual and normalize
        X = self.LayerNorm1(self.dropout1(mha) + X)

        # Get feedforward
        X = self.Feedforward(X)

        # Add residual and normalize
        memory = self.LayerNorm2(X + self.dropout2(X))

        return memory 

# Define Encoder class
class Encoder(nn.Module):
    def __init__(self, T, d_model, nhead, d_ff, num_layers,dk,dv, dropout=0.1):
        super().__init__()

        # Define encoder layers
        self.encoders = nn.ModuleList([EncoderLayer(T, d_model, nhead, d_ff, dk, dv, dropout) for _ in range(num_layers)])

    def forward(self, X, mask=None):

        # Pass through each encoder layer
        for encoder in self.encoders:
            X = encoder(X, mask)

        return X



class lstransformer(nn.Module):
        def __init__(self, his_size, d_model, ffdim, nhead, num_layers, user_vocab_size, attention_dim, word_emb_dim, filter_num, window_size, word_vectors ,device,dropout=0.2):
            super().__init__()
            self.num_layers = num_layers

            # Positional encoding
            self.positional_encoding = PositionalEncoding(his_size,d_model)

            # Encoder 
            self.encoder = Encoder(his_size, d_model, nhead, ffdim, num_layers,dk=256,dv=512, dropout=dropout)
            
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
            memory = self.encoder(encoded_his, mask = his_key_mask)
            

            # dropouts
            memory = self.dropout2(memory)

            # Embed candidates
            embed_cand = th.empty((candidates.shape[0],candidates.shape[1],400),device=self.device)
            for i in range(candidates.shape[0]):
                embed_cand[i] = self.newsencoder(candidates[i])

            #Embed user id
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
