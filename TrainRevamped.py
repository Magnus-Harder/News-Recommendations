#%%
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
user_file_train = 'Data/MINDdemo_train/behaviors.tsv'
news_file_train = 'Data/MINDdemo_train/news.tsv'
word_dict_file = 'Data/MINDdemo_utils/word_dict_all.pkl'

user_file_test = 'Data/MINDdemo_dev/behaviors.tsv'
news_file_test = 'Data/MINDdemo_dev/news.tsv'



TrainData = NewsDataset(user_file_train, news_file_train, word_dict_file,train=True,device = device,
                        max_title_length=hparams['data']['title_size'],
                        max_history_length=hparams['data']['his_size'],
                        )

TrainDataLoader = DataLoader(TrainData, batch_size=32, shuffle=False)


TestData = NewsDataset(user_file_test, news_file_test, word_dict_file,train=False, device = device,
                        max_title_length=hparams['data']['title_size'],
                        max_history_length=hparams['data']['his_size'],
                        userid_dict=TrainData.userid_dict,
                        )

ValiDataLoader = DataLoader(TestData, batch_size=1, shuffle=False)


#%%
# Import Model
from Models.LSTURClone import LSTURini

# Set Model Architecture
LSTUR_con_module = LSTURini(
    attention_dim = hparams['model']['attention_hidden_dim'],
    word_emb_dim = hparams['model']['word_emb_dim'],
    dropout = hparams['model']['dropout'],
    filter_num = hparams['model']['filter_num'],
    windows_size = hparams['model']['window_size'],
    gru_unit = hparams['model']['gru_unit'],
    user_size = TrainData.userid_dict.__len__(),
    word_vectors = word_embedding,
    device = device
)

# Move to device
model = LSTUR_con_module.to(device)

# Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=hparams['train']['learning_rate'])

# Define Loss
def loss_fn(Scores,n_positive):
    n = Scores.shape[0]

    loss = 0
    for i in range(n):
        loss += -th.log(th.exp(Scores[i,:n_positive[i],0])/th.exp(Scores[i,:n_positive[i],:]).sum(dim=1)).sum()

    return loss/n

def loss_fn_vali(Scores,labels):

    loss = -th.log(th.exp(Scores[labels == 1].sum())/th.exp(Scores).sum())

    return loss

#%%
# Pre Training Validation step
model.train(False)

softmax = th.nn.Softmax(dim=1)

with th.no_grad():
    
    # Initialize variables
    AUC_pre= 0
    MRR_pre= 0
    loss_pre = 0

    # Load validation data
    DataIterators = iter(ValiDataLoader)

    # Loop through validation data
    N_vali = len(ValiDataLoader)
    for _ in range(N_vali):

        # Load batch
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = next(DataIterators)

        # Get length of impression
        idx = impressions_length.item()

        # Get output
        Scores = model(user_id, history_title, history_length, impressions_title[:idx], n_positive)
        pred = softmax(Scores)[0]
        #print(pred)
        # Calculate loss
        loss = loss_fn_vali(Scores[0],labels[0,:idx])
        loss_pre += loss.item()/N_vali


        # # Calculate metrics
        AUC_score = ValidateModel.ROC_AUC(pred.detach().cpu(), labels[0,:idx].detach().cpu())
        #MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach().cpu(), pred.detach().cpu()[0])

        AUC_pre += AUC_score/N_vali
        #MRR_pre += MRR_score.item()/N_vali


print(f"Pre Training AUC: {AUC_pre}, MRR: {MRR_pre}, Loss: {loss_pre}")

#%%

# Initialize lists for saving metrics
AUC = [AUC_pre]
MRR = [MRR_pre]
losses = []
loss_vali = [loss_pre]

epochs = 5
batches = len(TrainDataLoader)//32

# Training loop
for epoch in range(epochs):

    # Training step
    model.train(True)
    optimizer.zero_grad()

    TestIterator = iter(TrainDataLoader)
    # Loop through training data
    for _ in range(batches):

        # Load batch
        user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = next(iter(TestIterator))

        # Get output
        Scores = model(user_id, history_title, history_length, impressions_title, n_positive)

        # Calculate loss
        loss = loss_fn(Scores,n_positive)
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        # Save loss   
        losses.append(loss.item())

    # Validation step
    model.train(False)    
    
    # No gradient calculation
    with th.no_grad():
    
        # Initialize variables
        AUC_epoch = 0
        MRR_epoch = 0
        loss_vali_epoch = 0
        # Load validation data
        DataIterators = iter(ValiDataLoader)

        # Loop through validation data
        for _ in range(N_vali):

            # Load batch
            user_id, history_title, history_abstract, history_length, impressions_title, impressions_abstract, impressions_length, labels, n_positive = next(DataIterators)

            # Get length of impression
            idx = impressions_length.item()

            # Get output
            Scores = model(user_id, history_title, history_length, impressions_title[:idx], n_positive)
            pred = softmax(Scores)[0]
  
            # Calculate loss
            loss = loss_fn_vali(Scores[0],labels[0,:idx])
            loss_vali_epoch += loss.item()/N_vali


            # # Calculate metrics
            AUC_score = ValidateModel.ROC_AUC(pred.detach().cpu(), labels[0,:idx].detach().cpu())
            #MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach().cpu(), pred.detach().cpu()[0])

            AUC_epoch += AUC_score/N_vali
            #MRR_pre += MRR_score.item()/N_vali

            # Save loss, AUC and MRR
            loss_vali.append(loss_vali_epoch)
            AUC.append(AUC_epoch)
            MRR.append(MRR_epoch)


    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(f"AUC: {AUC_epoch}. MRR: {MRR_epoch}. Loss: {loss_vali_epoch}.")

# Saving Training Log
with open('Revamped.pkl', 'wb') as f:
    pkl.dump([AUC,MRR,losses,loss_vali], f)







# %%
