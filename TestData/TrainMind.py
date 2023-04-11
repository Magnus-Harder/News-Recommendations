#%%
import tensorflow as tf

from TestData.MindDependencies.MindIt import MINDIterator
from TestData.MindDependencies.Utils import get_mind_data_set

from tqdm import tqdm
import pickle as pkl

from General.Utils import ValidateModel
from DataIterator import NewsDataset
from torch.utils.data import DataLoader


import torch as th
import numpy as np
import yaml

# Import Hparam
with open('Data/MINDdemo_utils/lstur.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')


#%%
# Define Device
device = 'cuda' if th.cuda.is_available() else 'cpu'

# Define Data, Dataset and DataLoaders
train_behaviors_file = 'Data/MINDdemo_train/behaviors.tsv'
train_news_file = 'Data/MINDdemo_train/news.tsv'
word_dict_file = 'Data/MINDdemo_utils/word_dict_all.pkl'
user_dict_file = 'Data/MINDdemo_utils/uid2index.pkl'

valid_behaviors_file = 'Data/MINDdemo_dev/behaviors.tsv'
valid_news_file = 'Data/MINDdemo_dev/news.tsv'


#%%
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

train_iterator = MINDIterator(hparamsdata,npratio=4)
test_iterator = MINDIterator(hparamsdata)

batch_loader_train = train_iterator.load_data_from_file(train_news_file, train_behaviors_file)
batch_loader_valid = test_iterator.load_data_from_file(valid_news_file, valid_behaviors_file)

for batch in batch_loader_valid:
    continue


#%%
from TestData.LSTURMind import LSTURini


# Set Model Architecture
LSTUR_con_module = LSTURini(
    attention_dim = hparams['model']['attention_hidden_dim'],
    word_emb_dim = hparams['model']['word_emb_dim'],
    dropout = hparams['model']['dropout'],
    filter_num = hparams['model']['filter_num'],
    windows_size = hparams['model']['window_size'],
    gru_unit = hparams['model']['gru_unit'],
    user_size = train_iterator.uid2index.__len__() + 1,
    word_vectors = word_embedding,
    device = device
)



model = LSTUR_con_module.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

loss_fn = th.nn.CrossEntropyLoss()

# Define Loss
# def loss_fn(Scores,n_positive):
#     n = Scores.shape[0]

#     loss = 0
#     for i in range(n):
#         loss += -th.log(th.exp(Scores[i,:n_positive[i],0])/th.exp(Scores[i,:n_positive[i],:]).sum(dim=1)).sum()

#     return loss/n

def loss_fn_vali(Scores,labels):

    loss = -th.log(th.exp(Scores[labels == 1].sum())/th.exp(Scores).sum())

    return loss

def batch_to_tensor(batch):
    user_id = th.from_numpy(batch['user_index_batch'])
    history_title = th.from_numpy(batch['clicked_title_batch'])
    impressions_title = th.from_numpy(batch['candidate_title_batch'])
    labels = th.from_numpy(batch['labels'])

    return user_id, history_title, impressions_title, labels


# Pre Training Validation step
model.train(False)

softmax = th.nn.Softmax(dim=1)
preds = {i : [] for i in test_iterator.impr_indexes}
labels_dc = {i : [] for i in test_iterator.impr_indexes}

with th.no_grad():
    
    # Initialize variables
    AUC_pre= 0
    MRR_pre= 0
    loss_pre = 0

    # Load validation data
    i = 0


    # Loop through validation data)
    for batch in tqdm(batch_loader_valid):
        i += 1

        # Load batch
        user_id, history_title, impressions_title, labels = batch_to_tensor(batch)

        Scores = model(user_id.flatten(), history_title, impressions_title)

        for idx, id in enumerate(batch['impression_index_batch']):
            preds[id.item()].append( Scores[idx].item())
            labels_dc[id.item()].append(labels[idx].item())

        pred = softmax(Scores)
        #print(pred)
        # Calculate loss
        #loss = loss_fn_vali(Scores,labels)
        #loss_pre += loss.item()


        # # Calculate metrics
        #AUC_score = ValidateModel.ROC_AUC(Scores.detach().cpu(), labels.detach().cpu())
        #MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach().cpu(), pred.detach().cpu()[0])

        #AUC_pre += AUC_score
        #MRR_pre += MRR_score.item()/N_vali
    
    for key in preds.keys():
        # preds[key] = np.array(preds[key])
        # preds[key] = preds[key].argsort()[::-1]
        # preds[key] = np.where(preds[key] == 0)[0][0]
        # preds[key] = 1/(preds[key] + 1)
        # MRR_pre += preds[key]
        if len(preds[key]) == 0:
            continue
        if not 1 in labels_dc[key]:
            continue

        AUC_pre += ValidateModel.ROC_AUC(preds[key], labels_dc[key])
        loss_pre += loss_fn_vali(th.tensor(preds[key]), th.tensor(labels_dc[key]))
        #MRR_pre += ValidateModel.mean_reciprocal_rank(th.tensor(labels_dc[key]), th.tensor(preds[key]))

    # MRR_pre = MRR_pre/len(preds.keys())

    # Calculate average metrics
    AUC_pre = AUC_pre/i
    #MRR_pre = MRR_pre/i
    loss_pre = loss_pre/i

print(f"Pre Training AUC: {AUC_pre}, MRR: {MRR_pre}, Loss: {loss_pre}")

# %%
