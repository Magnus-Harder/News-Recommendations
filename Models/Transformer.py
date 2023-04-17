#%%
import torch as th
from torch import nn
import math
from TestData.LSTURMind import NewsEncoder

#%%
class lstransformer(nn.Module):
        def __init__(self, his_size, candidate_size ,d_model, ffdim, nhead, num_layers, newsencoder, dropout=0.1):
            super().__init__()


            # Encoder and decoder
            self.encoderlayer = nn.TransformerEncoderLayer(d_model,nhead, dim_feedforward=ffdim, dropout=dropout)
            self.decoderlayer = nn.TransformerDecoderLayer(d_model,nhead, dim_feedforward=ffdim, dropout=dropout)

            self.encoder = nn.TransformerEncoder(encoder_layer = self.encoderlayer,
                                                     num_layers = num_layers
                                                     )

            self.decoder = nn.TransformerDecoder(decoder_layer = self.decoderlayer,
                                                 num_layers = num_layers)
            

            #Newsencoder and userembedder
            self.newsencoder = newsencoder
            self.UserEmbedding = nn.Embedding(his_size, d_model ,padding_idx=0)


            #Final linear layer
            self.outlayer = nn.Linear(d_model, candidate_size)
            self.softmax = nn.Softmax(candidate_size)

        def forward(self, embed_his, user_id, candidates):
            
            memory = self.encoder(embed_his)
            # Add user embedding to memory
            memory = memory + self.UserEmbedding(user_id)

            #Decode candidates with memory
            decoded = self.decoder(candidates,memory,)
            
            #Final layer
            out = self.outlayer(decoded)
            out = self.softmax(out)

            return out








