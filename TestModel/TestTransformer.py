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

dataset = "demo"

# Import Hparam
with open('hparams/Transformerhparam.yaml','r') as stream:
    hparams = yaml.safe_load(stream)


# Import word_vec
word_embedding = np.load(f'Data/MIND{dataset}_utils/embedding_all.npy')
word_embedding = word_embedding.astype(np.float32)

#th.manual_seed(2021)

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
    word_dict = pkl.load(f)
with open (f"Data/MIND{dataset}_utils/uid2index.pkl", "rb") as f:
    uid2index = pkl.load(f)


TestData = NewsDataset(valid_behaviors_file, valid_news_file, word_dict_file, userid_dict=uid2index, device=device, train=False)





#%%
# Import Model
print('Importing Model...', hparams['model']['Transformer']['model'])
if hparams['model']['Transformer']['model'] == 'Additive':
    from ModelsTransformer.TransformerAdditive import lstransformer
elif hparams['model']['Transformer']['model'] == 'Ini':
    from ModelsTransformer.TransformerIni import lstransformer
elif hparams['model']['Transformer']['model'] == 'IniDot':
    from ModelsTransformer.TransformerIniDot import lstransformer
elif hparams['model']['Transformer']['model'] == 'NoMem':
    from ModelsTransformer.TransformernoMem import lstransformer
elif hparams['model']['Transformer']['model'] == 'UserEmb':
    from ModelsTransformer.TransformerUserEmb import lstransformer
    hparamsdata.his_size += 1
elif hparams['model']['Transformer']['model'] == 'UserEmbDot':
    from ModelsTransformer.TransformerUserEmbDot import lstransformer
    hparamsdata.his_size += 1
elif hparams['model']['Transformer']['model'] == 'IniOwn':
    from ModelsTransformer.TransformerIniOwn import lstransformer
elif hparams['model']['Transformer']['model'] == 'IniLSTUR':
    from ModelsTransformer.TransformerIniLSTUR import lstransformer

# Print Specefic Hparams
print('nheads:', hparams['model']['Transformer']['num_heads'])
print('num_layers:', hparams['model']['Transformer']['num_layers'])
print('dff:', hparams['model']['Transformer']['dff'])
print('Hparam set', hparams['model']['Transformer']['set'])

#%%
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

loss_fn = th.nn.BCELoss()
loss_vali = th.nn.BCELoss()

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