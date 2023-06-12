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
from General.DataIterator import NewsDataset
from torch.utils.data import DataLoader
from General.MindDependencies.Metrics import cal_metric

dataset = "small"

# Import Hparam
with open('hparams/Transformerhparam.yaml','r') as stream:
    hparams = yaml.safe_load(stream)

if dataset == "small":
    pass
else:
    # Import word_vec
    word_embedding = np.load(f'Data/MIND{dataset}_utils/embedding_all.npy')
    word_embedding = word_embedding.astype(np.float32)

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
    word_dict = pkl.load(f)
with open (f"Data/MIND{dataset}_utils/uid2index.pkl", "rb") as f:
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

# Define Dataset
TrainData = NewsDataset(train_behaviors_file, train_news_file, word_dict_file, userid_dict=None,npratio=hparams['data']['npratio'], device=device,train=True,transformer=True)
TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=TrainData.userid_dict, device=device, train=False)

# Define word_embedding if dataset is small
if dataset == "small":
    word_dict = TrainData.word_dict

    # Import Glove
    import torchtext.vocab as vocab
    vec = vocab.GloVe(name='6B', dim=300, cache='torchtext_data6B')
    word_embedding = np.zeros((word_dict.__len__() + 1, 300))

    # Get word embedding
    for word, index in word_dict.items():
        if word in vec.stoi:
            word_embedding[index] = vec[word]
        else:
            word_embedding[index] = np.random.normal(scale=0.1, size=(300,))

    word_embedding = word_embedding.astype(np.float32)

#%%
# Import Model
print('Importing Model...', hparams['model']['Transformer']['model'])
if hparams['model']['Transformer']['model'] == 'Additive':
    from ModelsTransformer.TransformerAdditive import lstransformer
elif hparams['model']['Transformer']['model'] == 'Ini':
    from ModelsTransformer.TransformerIni import lstransformer
elif hparams['model']['Transformer']['model'] == 'IniLSTUR':
    from ModelsTransformer.TransformerIniLSTUR import lstransformer
elif hparams['model']['Transformer']['model'] == 'Decoder':
    from ModelsTransformer.TransformerDecoder import lstransformer
elif hparams['model']['Transformer']['model'] == 'Encoder':
    from ModelsTransformer.TransformerEncoder import lstransformer
    hparamsdata.his_size += 1

# Print Specefic Hparams
print('nheads:', hparams['model']['Transformer']['num_heads'])
print('num_layers:', hparams['model']['Transformer']['num_layers'])
print('dff:', hparams['model']['Transformer']['dff'])
print('Hparam set', hparams['model']['Transformer']['set'])

#%%
# Define Model
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

# Define mask for padding
def get_mask_key(batch_size,data_length, actual_length,device='cpu'):

    mask = th.zeros((batch_size,data_length),dtype=th.bool,device=device)

    if hparams['model']['Transformer']['model'] == 'UserEmb':
        actual_length += 1
              
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

    # Define DataLoader
    vali_batch_loader = DataLoader(TestData, batch_size=batch_size_vali, shuffle=False)

    # Iterate over DataLoader
    for batch in tqdm(vali_batch_loader):

        # unpack batch
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, _ = batch

        # Get batch size
        batch_size = user_id.shape[0]

        # Get mask
        history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length,device=device)

        # Get Scores
        Scores = model(user_id, history_title, history_mask, impressions_title)
        Scores = Scores.squeeze(-1)

        # Get loss and prediction
        for i in range(batch_size_vali):
            loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
            loss_vali.append(loss.item())
            labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
            preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
        
    # Get Metrics
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

# Train the model
for epoch in range(hparams['train']['epochs']):

    # Set model to train mode
    model.train(True)

    # Set batch size
    batch_size_train = hparamsdata.batch_size

    # Define Train DataLoader with shuffle
    train_data_loader = DataLoader(TrainData, batch_size=batch_size_train, shuffle=True)

    # Iterate over Trainin DataLoader
    for batch in tqdm(train_data_loader):

        # Zero the gradients
        optimizer.zero_grad()
        
        # Unpack batch
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels,n_positive = batch

        # Get batch size
        batch_size = user_id.shape[0]

        # Get mask
        history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)

        # Move to device
        user_id = user_id.to(device)
        history_title = history_title.to(device)
        history_length = history_length.to(device)
        impressions_title = impressions_title.to(device)
        labels = labels.to(device)
        history_mask = history_mask.to(device)

        # Get Scores
        Scores = model(user_id, history_title, history_mask, impressions_title)

        # Get loss and backpropagate
        loss = loss_fn(Scores, labels.argmax(dim=1).reshape(-1,1))
        loss.backward()
        optimizer.step()

        # Append loss to Evaluation dict
        Evaluation_dict['Loss_training'].append(loss.item())
    
    # Validation step
    with th.no_grad():
        model.eval()
        model.train(False)
        labels_all = []
        preds_all = []
        user_id_all = []
        loss_vali = []

        # Set batch size for validation
        batch_size_vali = 1

        # Define Validation DataLoader
        vali_batch_loader = DataLoader(TestData, batch_size=batch_size_vali, shuffle=False)

        # Iterate over Validation DataLoader
        for batch in tqdm(vali_batch_loader):

            # Unpack batch
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, _ = batch

            # Get batch size
            batch_size = user_id.shape[0]

            # Get mask
            history_mask = get_mask_key(batch_size,hparamsdata.his_size, history_length)

            # Move to device
            user_id = user_id.to(device)
            history_title = history_title.to(device)
            history_length = history_length.to(device)
            impressions_title = impressions_title.to(device)
            labels = labels.to(device)
            history_mask = history_mask.to(device)

            # Get Scores
            Scores = model(user_id, history_title,history_mask, impressions_title)
            Scores = Scores.squeeze(-1)

            # Get loss and prediction
            for i in range(batch_size_vali):
                loss = loss_fn(Scores[i,:impressions_length[i].item()], labels[i,:impressions_length[i].item()])
                loss_vali.append(loss.item())
                labels_all.append(labels[i,:impressions_length[i].item()].cpu().numpy())
                preds_all.append(Scores[i,:impressions_length[i].item()].detach().cpu().numpy())
                user_id_all.append(user_id.cpu().squeeze(0).numpy())

 
        # Get Metrics and append to Evaluation dict
        result = cal_metric(labels_all,preds_all,metrics=['group_auc', 'mean_mrr', 'ndcg@5;10'])
        result['loss'] = np.mean(loss_vali)
        
        # Append to Evaluation dict
        Evaluation_dict['AUC'].append(result['group_auc'])
        Evaluation_dict['MRR'].append(result['mean_mrr'])
        Evaluation_dict['NDCG5'].append(result['ndcg@5'])
        Evaluation_dict['NDCG10'].append(result['ndcg@10'])
        Evaluation_dict['loss_vali'].append(result['loss'])

    # Print results
    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(result)

# %%

# Saving Training Logs
filestring = f'EvaluationTranformer{hparams["model"]["Transformer"]["model"]}{hparams["model"]["Transformer"]["set"]}.pkl'
with open(filestring, 'wb') as f:
    pkl.dump([Evaluation_dict], f)

#%%
# Save Last Predictions
Dictfilestring = f'Transformer{hparams["model"]["Transformer"]["model"]}{dataset}Predictions.pkl'

with open(Dictfilestring, 'wb') as f:
    pkl.dump({'preds': preds_all, 'labels': labels_all, 'user ids': user_id_all, 'UserDict': TrainData.userid_dict}, f)

# %%
