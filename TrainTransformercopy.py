#%%
from tqdm import tqdm
import pickle as pkl

from General.Utils import ValidateModel
from DataIteratorTransformer import NewsDataset
from DataIterator import NewsDataset as NewsDatasetOld
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

TrainData = NewsDatasetOld(train_behaviors_file, train_news_file, word_dict_file, userid_dict=uid2index, train=True,transformer=True)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=uid2index)



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


TransformerModule = lstransformer(his_size = hparamsdata.his_size, 
                                  candidate_size = impressions_length,
                                  d_model = 400, 
                                  ffdim = 800, 
                                  nhead = 1, 
                                  num_layers = 3, 
                                  newsencoder = newsencoder,
                                  user_vocab_size=uid2index.__len__() + 1,
                                  device=device,
                                  dropout=0.2,
                                )

# Move to device
model = TransformerModule.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

# Define Loss
loss_fn = th.nn.CrossEntropyLoss()
loss_vali = th.nn.BCELoss()

def get_mask_key(batch_size,data_length, actual_length):

    mask = th.zeros((batch_size,data_length))


    for _ in range(batch_size):
        mask[_,actual_length[_]:] = 1

    return mask.bool()


#%%
# Pre Training Validation step
with th.no_grad():
    model.eval()
    model.train(False)
    labels_all = []
    preds_all = []
    loss_vali = []

    batch_size_vali = 1

    vali_batch_loader = DataLoader(TestData, batch_size=batch_size_vali, shuffle=False)
    for batch in tqdm(vali_batch_loader):
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels = batch

        batch_size = user_id.shape[0]

        history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)

        user_id = user_id.to(device)
        history_title = history_title.to(device)
        history_length = history_length.to(device)
        impressions_title = impressions_title.to(device)
        labels = labels.to(device)
        history_mask = history_mask.to(device)


        Scores = model(user_id, history_title, None,history_mask, impressions_title, None, None)
        Scores = Scores.squeeze(-1)

        for i in range(batch_size_vali):

            loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
            loss_vali.append(loss.item())

            labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
            preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
        

    Pre_training = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
    Pre_training['loss'] = np.mean(loss_vali)

    print(Pre_training)
#%%


# %%
# Train the model
AUC = [Pre_training['group_auc']]
MRR = [Pre_training['mean_mrr']]
NDCG5 = [Pre_training['ndcg@5']]
NDCG10 = [Pre_training['ndcg@10']]
Loss_vali = [Pre_training['loss']]
Loss_training = []


for epoch in range(5):
    model.train(True)

    batch_size_train = hparamsdata.batch_size

    train_data_loader = DataLoader(TrainData, batch_size=hparamsdata.batch_size, shuffle=True)


    for batch in tqdm(train_data_loader):
        optimizer.zero_grad()
            
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels,n_positive = batch

        batch_size = user_id.shape[0]

        history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)


        user_id = user_id.to(device)
        history_title = history_title.to(device)
        history_length = history_length.to(device)
        impressions_title = impressions_title.to(device)
        labels = labels.to(device)
        history_mask = history_mask.to(device)

        Scores = model(user_id, history_title, None,history_mask, impressions_title, None, None)

        loss = loss_fn(Scores, labels.argmax(dim=1).reshape(-1,1))

        loss.backward()

        optimizer.step()


    # Validation step
    with th.no_grad():
        model.eval()
        model.train(False)
        labels_all = []
        preds_all = []
        loss_vali = []

        batch_size_vali = 1

        vali_batch_loader = DataLoader(TestData, batch_size=batch_size_vali, shuffle=False)
        for batch in tqdm(vali_batch_loader):
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels = batch

            batch_size = user_id.shape[0]

            history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)

            user_id = user_id.to(device)
            history_title = history_title.to(device)
            history_length = history_length.to(device)
            impressions_title = impressions_title.to(device)
            labels = labels.to(device)
            history_mask = history_mask.to(device)


            Scores = model(user_id, history_title, None,history_mask, impressions_title, None, None)
            Scores = Scores.squeeze(-1)

            for i in range(batch_size_vali):

                loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
                loss_vali.append(loss.item())

                labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
                preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
            

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