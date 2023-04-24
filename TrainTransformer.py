#%%
from tqdm import tqdm
import pickle as pkl

from General.Utils import ValidateModel
from DataIterator import NewsDataset
from torch.utils.data import DataLoader


from TestData.MindDependencies.Metrics import cal_metric


import torch as th
import numpy as np
import yaml

# Import Hparam
with open('Data/MINDdemo_utils/lstur.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')
word_embedding = word_embedding.astype(np.float32)


# %%
# Define Device
device = 'cuda' if th.cuda.is_available() else 'cpu'

# Define Data, Dataset and DataLoaders
train_behaviors_file = 'Data/MINDdemo_train/behaviors.tsv'
train_news_file = 'Data/MINDdemo_train/news.tsv'
word_dict_file = 'Data/MINDdemo_utils/word_dict_all.pkl'
user_dict_file = 'Data/MINDdemo_utils/uid2index.pkl'

valid_behaviors_file = 'Data/MINDdemo_dev/behaviors.tsv'
valid_news_file = 'Data/MINDdemo_dev/news.tsv'

# %%
import pickle

with open ("Data/MINDdemo_utils/word_dict.pkl", "rb") as f:
    word_dict = pickle.load(f)
with open ("Data/MINDdemo_utils/uid2index.pkl", "rb") as f:
    uid2index = pickle.load(f)

from dataclasses import dataclass

@dataclass
class HyperParams:
    batch_size: int
    title_size: int
    his_size: int
    wordDict_file: str
    userDict_file: str

hparamsdata = HyperParams(
    batch_size=32,
    title_size=20,
    his_size=50,
    wordDict_file=word_dict_file,
    userDict_file=user_dict_file,
)

TrainData = NewsDataset(train_behaviors_file, train_news_file, word_dict_file, userid_dict=uid2index, train=True)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=uid2index, train=False)



from TestData.LSTURMind import NewsEncoder
newsencoder = NewsEncoder(attention_dim = hparams['model']['attention_hidden_dim'],
                        word_emb_dim = hparams['model']['word_emb_dim'],
                        dropout = hparams['model']['dropout'],
                        filter_num = hparams['model']['filter_num'],
                        windows_size = hparams['model']['window_size'],
                        gru_unit = hparams['model']['gru_unit'],
                        word_vectors = word_embedding,
                        device = device
                        )   




#%%
# Import Model
from Models.Transformer import lstransformer
impressions_length = 50


TransformerModule = lstransformer(his_size = 50, 
                                  candidate_size = impressions_length,
                                  d_model = 400, 
                                  ffdim = 100, 
                                  nhead = 1, 
                                  num_layers = 1, 
                                  newsencoder = newsencoder,
                                  user_vocab_size=uid2index.__len__() + 1,
                                  dropout=0.1,
                                )

# Move to device
model = TransformerModule.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

# Define Loss
loss_fn = th.nn.CrossEntropyLoss()

#%%
# Pre Training Validation step
with th.no_grad():
    model.eval()
    model.train(False)
    labels_all = []
    preds_all = []
    loss_vali = []

    vali_batch_loader = DataLoader(TestData, batch_size=1, shuffle=False)
    i = 0
    for batch in tqdm(vali_batch_loader):
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

        user_id = user_id.to(device)
        history_title = history_title.to(device)
        history_length = history_length.to(device)
        impressions_title = impressions_title.to(device)
        labels = labels.to(device)

        Scores = model(user_id, history_title, impressions_title)
        Scores = Scores.squeeze(-1)

        if labels.shape[1] > 50:
            labels = labels[:, :50]
        elif labels.shape[1] < 50:
            labels = th.cat((labels, th.zeros((labels.shape[0], 50 - labels.shape[1])).to(device)), dim=1)


        loss = loss_fn(Scores, labels)
        loss_vali.append(loss.item())
        if sum(labels.squeeze(0).cpu().numpy()) == 0:
            continue
        labels_all.append(labels.squeeze(0).cpu().numpy())
        preds_all.append(Scores.squeeze(0).detach().cpu().numpy())
        i += 1
        if i > 100:
            break
#%%

Pre_training = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
Pre_training['loss'] = np.mean(loss_vali)

print(Pre_training)



# %%
# Train the model
AUC = [Pre_training['group_auc']]
MRR = [Pre_training['mean_mrr']]
NDCG5 = [Pre_training['ndcg@5']]
NDCG10 = [Pre_training['ndcg@10']]
Loss_vali = [Pre_training['loss']]
Loss_training = []


for epoch in range(1):
    model.train(True)

    train_data_loader = DataLoader(TrainData, batch_size=hparamsdata.batch_size, shuffle=True)
    i = 0
    for batch in tqdm(train_data_loader):

        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

        user_id = user_id.to(device)
        history_title = history_title.to(device)
        history_length = history_length.to(device)
        impressions_title = impressions_title.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        Scores = Scores.squeeze(-1)

        Scores = model(user_id, history_title, impressions_title)

        

        loss = loss_fn(Scores,labels)

        loss.backward()

        optimizer.step()

        Loss_training.append(loss.item())

        i += 1
        if i > 10:
            break
    break
    with th.no_grad():
        model.eval()
        model.train(False)
        labels_all = []
        preds_all = []
        loss_vali = []

        vali_batch_loader = DataLoader(TestData, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(vali_batch_loader):
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

            user_id = user_id.to(device)
            history_title = history_title.to(device)
            history_length = history_length.to(device)
            impressions_title = impressions_title.to(device)
            labels = labels.to(device)

            Scores = model(user_id, history_title, history_length, impressions_title)

            loss = loss_fn(Scores, labels)
            loss_vali.append(loss.item())
            if sum(labels.squeeze(0).cpu().numpy()) == 0:
                continue
        
            labels_all.append(labels.squeeze(0).cpu().numpy())
            preds_all.append(Scores.squeeze(0).detach().cpu().numpy())
            i += 1
            if i > 100:
                break


        result = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
        result['loss'] = np.mean(loss_vali)

        AUC.append(result['group_auc'])
        MRR.append(result['mean_mrr'])
        NDCG5.append(result['ndcg@5'])
        NDCG10.append(result['ndcg@10'])
        Loss_vali.append(result['loss'])
    
    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(result)

# %%

# Saving Training Logs
with open('TrainTransformer.pkl', 'wb') as f:
    pickle.dump([Loss_training,AUC,MRR,NDCG5,NDCG10,Loss_vali], f)