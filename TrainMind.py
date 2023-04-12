#%%
# Load from Scripts
from TestData.MindDependencies.MindIt import MINDIterator
from TestData.MindDependencies.Utils import get_mind_data_set, validate_model


from tqdm import tqdm
import torch as th
import numpy as np
import yaml
import tensorflow as tf
import pickle

# Import Hparam
with open('Data/MINDdemo_utils/lstur.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')
word_embedding = word_embedding.astype(np.float32)

hparamstrain = hparams['train']
hparamsmodel = hparams['model']

# %%
# Define Device
device = 'cuda' if th.cuda.is_available() else 'mps'

# Define Data, Dataset and DataLoaders
train_behaviors_file = 'Data/MINDdemo_train/behaviors.tsv'
train_news_file = 'Data/MINDdemo_train/news.tsv'
word_dict_file = 'Data/MINDdemo_utils/word_dict_all.pkl'
user_dict_file = 'Data/MINDdemo_utils/uid2index.pkl'

valid_behaviors_file = 'Data/MINDdemo_dev/behaviors.tsv'
valid_news_file = 'Data/MINDdemo_dev/news.tsv'



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
    batch_size=hparamstrain['batch_size'],
    title_size=hparams['data']['title_size'],
    his_size=hparams['data']['his_size'],
    wordDict_file=word_dict_file,
    userDict_file=user_dict_file,
)

train_iterator = MINDIterator(hparamsdata,npratio=hparams['data']['npratio'])
test_iterator = MINDIterator(hparamsdata)

# %%
from TestData.LSTURMind import LSTURini

# Set Model Architecture
LSTUR_con_module = LSTURini(
    attention_dim = hparamsmodel['attention_hidden_dim'],
    word_emb_dim = hparamsmodel['word_emb_dim'],
    dropout = hparamsmodel['dropout'],
    filter_num = hparamsmodel['filter_num'],
    windows_size = hparamsmodel['window_size'],
    gru_unit = hparamsmodel['gru_unit'],
    user_size = train_iterator.uid2index.__len__() + 1,
    word_vectors = word_embedding,
    device = device
)

# Training 
print(device)
"""

model = LSTUR_con_module.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=hparamstrain['learning_rate'])

loss_fn = th.nn.CrossEntropyLoss()

# Training
with th.no_grad():
	model.eval()
	model.train(False)
	Pre_training = validate_model(model, valid_news_file, valid_behaviors_file, test_iterator, device, metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])

def batch_to_tensor(batch, device):
    user_id = th.from_numpy(batch['user_index_batch']).to(device).flatten()
    history_title = th.from_numpy(batch['clicked_title_batch']).to(device)
    impressions_title = th.from_numpy(batch['candidate_title_batch']).to(device)
    labels = th.from_numpy(batch['labels']).to(device)

    return user_id, history_title, impressions_title, labels
#%%
# Train the model
AUC = [Pre_training['group_auc']]
MRR = [Pre_training['mean_mrr']]
NDCG5 = [Pre_training['ndcg@5']]
NDCG10 = [Pre_training['ndcg@10']]
training_loss = []

for epoch in range(hparamstrain['epochs']):
    optimizer.zero_grad()

    for batch in tqdm(train_iterator.load_data_from_file(train_news_file, train_behaviors_file)):

        user_id, history_title, impressions_title, labels = batch_to_tensor(batch,device)


        model.train()

        optimizer.zero_grad()

        Scores = model(user_id, history_title, impressions_title)

        loss = loss_fn(Scores,labels)

        loss.backward()

        optimizer.step()

        training_loss.append(loss.item())
    
    with th.no_grad():
        model.eval()
        model.train(False)

        result = validate_model(model, valid_news_file, valid_behaviors_file, test_iterator, device, metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])

        AUC.append(result['group_auc'])
        MRR.append(result['mean_mrr'])
        NDCG5.append(result['ndcg@5'])
        NDCG10.append(result['ndcg@510'])
    
    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(result)

# %%

# Saving Training Logs
with open('MindTrain.pkl', 'wb') as f:
    pickle.dump([training_loss,AUC,MRR,NDCG5,NDCG10], f)
"""
