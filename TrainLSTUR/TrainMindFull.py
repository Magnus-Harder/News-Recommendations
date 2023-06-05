#%%
# Load Packages
print('Loading Packages...')

from tqdm import tqdm
import torch as th
import numpy as np
import yaml
import pickle


# Add path
import sys
import os

print(os.getcwd())
sys.path.insert(1,os.getcwd())


#%%
# Load from Scripts
from DataLoaders.DataIterator import NewsDataset
from torch.utils.data import DataLoader
from TestData.MindDependencies.Metrics import cal_metric

dataset = 'small'

# Import Hparam
with open(f'Data/MIND{dataset}_utils/lstur.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

if dataset == 'demo':
    # Import word_vec
    word_embedding = np.load(f'Data/MIND{dataset}_utils/embedding_all.npy')
    word_embedding = word_embedding.astype(np.float32)

hparamstrain = hparams['train']
hparamsmodel = hparams['model']

# %%
# Define Device
device = 'cuda' if th.cuda.is_available() else 'cpu'

# Define Data, Dataset and DataLoaders
train_behaviors_file = f'Data/MIND{dataset}_train/behaviors.tsv'
train_news_file = f'Data/MIND{dataset}_train/news.tsv'
word_dict_file = f'Data/MIND{dataset}_utils/word_dict_all.pkl'
user_dict_file = f'Data/MIND{dataset}_utils/uid2index.pkl'

valid_behaviors_file = f'Data/MIND{dataset}_dev/behaviors.tsv'
valid_news_file = f'Data/MIND{dataset}_dev/news.tsv'


with open (f"Data/MIND{dataset}_utils/word_dict.pkl", "rb") as f:
    word_dict = pickle.load(f)
with open (f"Data/MIND{dataset}_utils/uid2index.pkl", "rb") as f:
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
    batch_size=hparamstrain['batch_size'],
    title_size=hparams['data']['title_size'],
    his_size=hparams['data']['his_size'],
    wordDict_file=word_dict_file,
    userDict_file=user_dict_file,
)

TrainData = NewsDataset(train_behaviors_file, train_news_file, word_dict_file, userid_dict=None,npratio=hparams['data']['npratio'],device = device, train=True,transformer=False)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=TrainData.userid_dict, train=False, device = device,transformer=False)

if dataset == "small":
    word_dict = TrainData.word_dict

    # Import Glove
    import torchtext.vocab as vocab

    vec = vocab.GloVe(name='6B', dim=300, cache='torchtext_data6B')

    word_embedding = np.zeros((word_dict.__len__() + 1, 300))

    for word, index in word_dict.items():
        if word in vec.stoi:
            word_embedding[index] = vec[word]
        else:
            word_embedding[index] = np.random.normal(scale=0.1, size=(300,))
    
    word_embedding = word_embedding.astype(np.float32)

# %%
from ModelsLSTUR.LSTURini import LSTURini

# Set Model Architecture
LSTURini_module = LSTURini(
    attention_dim = hparamsmodel['attention_hidden_dim'],
    word_emb_dim = hparamsmodel['word_emb_dim'],
    dropout = hparamsmodel['dropout'],
    filter_num = hparamsmodel['filter_num'],
    windows_size = hparamsmodel['window_size'],
    gru_unit = hparamsmodel['gru_unit'],
    user_size = uid2index.__len__() + 1,
    word_vectors = word_embedding,
    device = device
)

# Training 
print(device)

model = LSTURini_module.to(device)


# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=hparamstrain['learning_rate'])

loss_fn = th.nn.CrossEntropyLoss()

#%%
# Pretraining Evaluation
with th.no_grad():
    model.eval()
    model.train(False)
    labels_all = []
    preds_all = []
    loss_vali = []

    vali_batch_loader = DataLoader(TestData, batch_size=1, shuffle=False)

    for batch in tqdm(vali_batch_loader):
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

        Scores = model(user_id, history_title, history_length, impressions_title)

        loss = loss_fn(Scores, labels)
        loss_vali.append(loss.item())
    
        labels_all.append(labels.cpu().squeeze(0).numpy())
        preds_all.append(Scores.cpu().squeeze(0).detach().numpy())

    
    Pre_training = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
    Pre_training['loss'] = np.mean(loss_vali)
    
    print(Pre_training)



#%%
# Train the model
Evaluation_dict = {
    'AUC':[Pre_training['group_auc']],
    'MRR':[Pre_training['mean_mrr']],
    'NDCG5':[Pre_training['ndcg@5']],
    'NDCG10':[Pre_training['ndcg@10']],
    'loss_vali':[Pre_training['loss']],
    'Loss_training':[]
}


for epoch in range(hparams['train']['epochs']):
    model.train(True)

    train_data_loader = DataLoader(TrainData, batch_size=hparams['train']['batch_size'], shuffle=True)

    for batch in tqdm(train_data_loader):

        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

        Scores = model(user_id, history_title, history_length, impressions_title)

        loss = loss_fn(Scores,labels)

        loss.backward()

        optimizer.step()

        Evaluation_dict['Loss_training'].append(loss.item())
        optimizer.zero_grad()

    
    with th.no_grad():
        model.eval()
        model.train(False)
        labels_all = []
        preds_all = []
        loss_vali = []
        user_id_all = []

        vali_batch_loader = DataLoader(TestData, batch_size=1, shuffle=False)

        for batch in tqdm(vali_batch_loader):
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = batch

            Scores = model(user_id, history_title, history_length, impressions_title)

            loss = loss_fn(Scores, labels)
            loss_vali.append(loss.item())

            labels_all.append(labels.cpu().squeeze(0).numpy())
            preds_all.append(Scores.cpu().squeeze(0).detach().numpy())
            user_id_all.append(user_id.cpu().squeeze(0).numpy())
  
        
        result = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
        result['loss'] = np.mean(loss_vali)


        Evaluation_dict['AUC'].append(result['group_auc'])
        Evaluation_dict['MRR'].append(result['mean_mrr'])
        Evaluation_dict['NDCG5'].append(result['ndcg@5'])
        Evaluation_dict['NDCG10'].append(result['ndcg@10'])
        Evaluation_dict['loss_vali'].append(result['loss'])

    
    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(result)

# %%

# Saving Training Logs
with open('EvalLSTURIni.pkl', 'wb') as f:
    pickle.dump(Evaluation_dict, f)

# Saving the model

Dictfilestring = f'LSTURini{dataset}Predictions.pkl'


#th.save(model.state_dict(), filestring)

with open(Dictfilestring, 'wb') as f:
    pickle.dump({'preds': preds_all, 'labels': labels_all, 'user ids': user_id_all, 'UserDict': TrainData.userid_dict}, f)

# %%
