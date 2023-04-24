#%%
import torch as th
from torch import nn
import math
from TestData.LSTURMind import NewsEncoder

#%%
class lstransformer(nn.Module):
        def __init__(self, his_size, candidate_size ,d_model, ffdim, nhead, num_layers, newsencoder,user_vocab_size ,dropout=0.1):
            super().__init__()
            self.num_layers = num_layers

            # Encoder 
            self.encoderlayer = nn.TransformerEncoderLayer(d_model,nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer = self.encoderlayer, num_layers = self.num_layers)
            
            # Decoder
            self.decoderlayer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer = self.decoderlayer, num_layers = self.num_layers)
            

            #Newsencoder and userembedder
            self.newsencoder = newsencoder
            self.UserEmbedding = nn.Embedding(user_vocab_size, d_model ,padding_idx=0)


            #Final linear layer
            self.outlayer = nn.Linear(d_model, 1)
            self.softmax = nn.Softmax(1)

        def forward(self, user_id, embed_his,his_mask, candidates,cand_mask):

            # Encode history
            encoded_his = th.empty((embed_his.shape[0],embed_his.shape[1],400))

            for i in range(embed_his.shape[0]):
                 encoded_his[i] = self.newsencoder(embed_his[i])
            
            #embed_his = self.newsencoder(embed_his)
            memory = self.encoder(encoded_his, src_key_padding_mask = his_mask)
            users = self.UserEmbedding(user_id)
            # Add user embedding to memory
            for i in range(memory.shape[0]):
                memory[i] = memory[i] + users[i]
    

            # Type error bug fixing
            candidates = candidates

            embed_cand = th.empty((candidates.shape[0],candidates.shape[1],400))
            
            for i in range(candidates.shape[0]):
                embed_cand[i] = self.newsencoder(candidates[i])



            #Decode candidates with memory
            decoded = self.decoder(embed_cand,memory,tgt_key_padding_mask = cand_mask)
            
            #Final layer
            out = self.outlayer(decoded)
            out = self.softmax(out)

            return out









# %%
