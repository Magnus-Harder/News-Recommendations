#%%
# Import Packages
from tqdm import tqdm
import pickle as pkl
import torch as th
import numpy as np
import yaml

# Add path
import sys
import os

sys.path.insert(1,os.getcwd())

# Import Self-Defined Modules
from DataLoaders.DataIterator import NewsDataset
from torch.utils.data import DataLoader
from TestData.MindDependencies.Metrics import cal_metric


# Import Hparam
with open('hparams/Transformerhparam.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')

# Import word_vec
word_embedding = np.load('Data/MINDdemo_utils/embedding_all.npy')
word_embedding = word_embedding.astype(np.float32)

th.manual_seed(2021)

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


with open ("Data/MINDdemo_utils/word_dict.pkl", "rb") as f:
    word_dict = pkl.load(f)
with open ("Data/MINDdemo_utils/uid2index.pkl", "rb") as f:
    uid2index = pkl.load(f)

from dataclasses import dataclass

@dataclass
class HyperParams:
    batch_size: int
    title_size: int
    his_size: int
    wordDict_file: str
    userDict_file: str

hparamsdata = HyperParams(
    batch_size=hparams['train']['batch_size'],
    title_size=hparams['data']['title_length'],
    his_size=hparams['data']['history_length'],
    wordDict_file=word_dict_file,
    userDict_file=user_dict_file,
)

TrainData = NewsDataset(train_behaviors_file, train_news_file, word_dict_file, userid_dict=uid2index,npratio=hparams['data']['npratio'], device=device,train=True,transformer=True)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=uid2index, device=device, train=False)





#%%
# Import Model
if hparams['model']['Transformer']['model'] == 'Additive':
    from ModelsTransformer.TransformerAdditive import lstransformer
elif hparams['model']['Transformer']['model'] == 'Ini':
    from ModelsTransformer.TransformerIni import lstransformer


TransformerModule = lstransformer(his_size = hparamsdata.his_size, 
                                  d_model = hparams['model']['Transformer']['d_model'], 
                                  ffdim = hparams['model']['Transformer']['dff'], 
                                  nhead = hparams['model']['Transformer']['num_heads'], 
                                  num_layers = hparams['model']['Transformer']['num_layers'], 
                                  user_vocab_size=uid2index.__len__() + 1,
                                  attention_dim = hparams['model']['News Encoder']['attention_hidden_dim'],
                                  word_emb_dim = hparams['model']['News Encoder']['word_emb_dim'],
                                  filter_num = hparams['model']['News Encoder']['filter_num'],
                                  window_size = hparams['model']['News Encoder']['window_size'],
                                  word_vectors = word_embedding,                                
                                  device=device,
                                  dropout=hparams['model']['Transformer']['dropout'],
                                )

# Move to device
model = TransformerModule.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=hparams['train']['learning_rate'])

# Define Loss
loss_fn = th.nn.CrossEntropyLoss()
loss_vali = th.nn.BCELoss()

def get_mask_key(batch_size,data_length, actual_length,device='cpu'):

    mask = th.zeros((batch_size,data_length),dtype=th.bool,device=device)


    for _ in range(batch_size):
        mask[_,actual_length[_]:] = 1
    mask = mask.bool()

    return mask


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
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, _ = batch

        batch_size = user_id.shape[0]

        history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length,device=device)


        Scores = model(user_id, history_title, history_mask, impressions_title)
        Scores = Scores.squeeze(-1)

        for i in range(batch_size_vali):

            loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
            loss_vali.append(loss.item())

            labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
            preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
        


    Pre_training = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
    Pre_training['loss'] = np.mean(loss_vali)

    print(Pre_training)


# %%
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

    batch_size_train = hparamsdata.batch_size

    train_data_loader = DataLoader(TrainData, batch_size=batch_size_train, shuffle=True)


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

        Scores = model(user_id, history_title, history_mask, impressions_title)

        loss = loss_fn(Scores, labels.argmax(dim=1).reshape(-1,1))

        loss.backward()

        optimizer.step()

        Evaluation_dict['Loss_training'].append(loss.item())


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
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, _ = batch

            batch_size = user_id.shape[0]

            history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)

            user_id = user_id.to(device)
            history_title = history_title.to(device)
            history_length = history_length.to(device)
            impressions_title = impressions_title.to(device)
            labels = labels.to(device)
            history_mask = history_mask.to(device)


            Scores = model(user_id, history_title,history_mask, impressions_title)
            Scores = Scores.squeeze(-1)

            for i in range(batch_size_vali):

                loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
                loss_vali.append(loss.item())

                labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
                preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
            

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
with open('EvaluationTranformerAdditive.pkl', 'wb') as f:
    pkl.dump([result], f)
# %%
