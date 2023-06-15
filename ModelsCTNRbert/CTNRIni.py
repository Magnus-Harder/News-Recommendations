#%%
import torch as th
from torch import nn
from ModelsCTNRbert.NewsEncoderBert import NewsEncoder



class CTNR(nn.Module):
        def __init__(self, ffdim, nhead, num_layers, user_vocab_size, topic_dim, subtopic_dim, topic_size, subtopic_size, title_vectors, abstract_vectors, device,dropout=0.2):
            super().__init__()
            self.num_layers = num_layers
            self.device = device

            # Newsencoder and userembedder
            self.newsencoder = NewsEncoder(topic_dim,subtopic_dim,topic_size,subtopic_size, dropout, title_vectors, abstract_vectors, device=device)
            self.d_model = self.newsencoder.dim_news
            self.UserEmbedding = nn.Embedding(user_vocab_size, self.d_model ,padding_idx=0)


            # Encoder 
            self.encoderlayer = nn.TransformerEncoderLayer(self.d_model,nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer = self.encoderlayer, num_layers = self.num_layers)


            # Decoder
            self.decoderlayer = nn.TransformerDecoderLayer(self.d_model, nhead, dim_feedforward=ffdim, dropout=dropout,batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer = self.decoderlayer, num_layers = self.num_layers)
            

            #Final linear layer
            self.outlayer = nn.Linear(self.d_model, 1)
            self.softmax = nn.Softmax(1)


            # Dropout_layers
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)
            self.dropout5 = nn.Dropout(dropout)

        def forward(self, user_id, embed_his, topic, subtopic, candidates, candidate_topic, candidate_subtopic, his_key_mask):

            # Encode history
            encoded_his = th.empty((embed_his.shape[0],embed_his.shape[1],self.d_model),device=self.device)
            for i in range(embed_his.shape[0]):
                 encoded_his[i] = self.newsencoder(embed_his[i],topic[i],subtopic[i])
            encoded_his = self.dropout1(encoded_his)


            # Encode history with transformer
            memory = self.encoder(encoded_his, src_key_padding_mask = his_key_mask)
            memory = self.dropout2(memory)

            # Embed candidates
            embed_cand = th.empty((candidates.shape[0],candidates.shape[1],self.d_model),device=self.device)
            for i in range(candidates.shape[0]):
                embed_cand[i] = self.newsencoder(candidates[i], candidate_topic[i], candidate_subtopic[i])
            embed_cand = self.dropout3(embed_cand)

            
            # Embed user and add to candidates
            users = self.UserEmbedding(user_id)
            users = self.dropout4(users)
            User_embed_cant = th.cat((users.unsqueeze(1),embed_cand),dim=1)

            # Decode candidates
            decoded = self.decoder(User_embed_cant,memory)
            decoded = self.dropout4(decoded)
            
            #Final layer
            out = self.outlayer(decoded)

            return out[:,1:]









# %%
