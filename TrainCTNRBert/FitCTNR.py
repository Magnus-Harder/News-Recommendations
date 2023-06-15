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

# Define Device
device = 'cuda' if th.cuda.is_available() else 'mps'

# Import Self-Defined Modules
from General.DataIteratorBert import NewsDataset
from torch.utils.data import DataLoader
from General.MindDependencies.Metrics import cal_metric

#%%
# Load and define Dataset
dataset = "small"

# Import Hparam
with open('hparams/CTNRBert.yaml','r') as stream:
    hparams = yaml.safe_load(stream)


# Define Data, Dataset and DataLoaders
train_behaviors_file = f'Data/MIND{dataset}_train/behaviors.tsv'
train_news_file = f'Data/MIND{dataset}_train/news.tsv'

valid_behaviors_file = f'Data/MIND{dataset}_dev/behaviors.tsv'
valid_news_file = f'Data/MIND{dataset}_dev/news.tsv'

# Load News Vocab
with open('Bert/News_vocab.pkl', 'rb') as f:
    News_vocab = pkl.load(f)
    News_dict = News_vocab.get_stoi()


# Define Dataset
TrainData = NewsDataset(user_file=train_behaviors_file, news_file=train_news_file,
                        News_dict=News_dict, 
                        device=device,
                        transformer=True,
                        train=True)

TestData = NewsDataset(user_file=valid_behaviors_file, news_file=valid_news_file,
                        News_dict=News_dict,
                        Category_vocab=TrainData.Category_vocab,
                        Subcategory_vocab=TrainData.Subcategory_vocab,
                        userid_dict=TrainData.userid_dict,
                        train=False,
                        device=device)

#%%
# Load predefined title and abstract embeddings
with open('Bert/title_bert.pkl', 'rb') as f:
    title_rep = pkl.load(f)

with open('Bert/abstract_bert.pkl', 'rb') as f:
    abstract_rep = pkl.load(f)



#%%
# Import Model
from ModelsCTNRbert.CTNRIni import CTNR

# Print Specefic Hparams
print('nheads:', hparams['model']['Transformer']['num_heads'])
print('num_layers:', hparams['model']['Transformer']['num_layers'])
print('dff:', hparams['model']['Transformer']['dff'])
print('Hparam set', hparams['model']['Transformer']['set'])

#%%
# Define Model
TransformerModule = CTNR(
                        ffdim = hparams['model']['Transformer']['dff'], 
                        nhead = hparams['model']['Transformer']['num_heads'], 
                        num_layers = hparams['model']['Transformer']['num_layers'], 
                        user_vocab_size=TrainData.userid_dict.__len__()+1,
                        topic_dim = 16*3,
                        subtopic_dim = 16*3,
                        topic_size = TrainData.Category_vocab.__len__()+1,
                        subtopic_size = TrainData.Subcategory_vocab.__len__()+1,
                        title_vectors = title_rep,
                        abstract_vectors = abstract_rep,
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
        user_id, history, history_category, history_subcategory,  history_length, impressions, impressions_category, impressions_subcategory, impressions_length, labels = batch

        # Get batch size
        batch_size = user_id.shape[0]

        # Get mask
        history_mask = get_mask_key(batch_size,50, history_length,device=device)

        # Get Scores
        Scores = model(user_id, history, history_category, history_subcategory, impressions, impressions_category, impressions_subcategory, history_mask)
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
    batch_size_train = hparams['data']['batch_size']

    # Define Train DataLoader with shuffle
    train_data_loader = DataLoader(TrainData, batch_size=batch_size_train, shuffle=True)

    # Iterate over Trainin DataLoader
    for batch in tqdm(train_data_loader):

        # Zero the gradients
        optimizer.zero_grad()
        
        # Unpack batch
        user_id, history, history_category, history_subcategory,  history_length, impressions, impressions_category, impressions_subcategory, impressions_length, labels = batch

        # Get batch size
        batch_size = user_id.shape[0]

        # Get mask
        history_mask = get_mask_key(batch_size,50, history_length)

        # Get Scores
        Scores = model(user_id, history, history_category, history_subcategory, impressions, impressions_category, impressions_subcategory, history_mask)

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
            user_id, history, history_category, history_subcategory,  history_length, impressions, impressions_category, impressions_subcategory, impressions_length, labels = batch

            # Get batch size
            batch_size = user_id.shape[0]

            # Get mask
            history_mask = get_mask_key(batch_size,50, history_length)

            # Get Scores
            Scores = model(user_id, history, history_category, history_subcategory, impressions, impressions_category, impressions_subcategory, history_mask)
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
