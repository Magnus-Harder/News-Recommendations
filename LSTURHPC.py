#%%
# Script used to train LSTUR model on HPC in order to recreate results from LSTUR-paper.
# Hyperparameters used in paper:
#   Word embedding dim = 300
#   Filters in CNN = 300
#   Windowsize CNN = 3
#   Dropout rate = 0.2
#   Masking probability = 0.5
#   Adam optimizer learning rate = 0.01
#   Batch size = 400
#   Negative samples pr positive sample = 4
#   Epochs = 10
#   History length = 50
#   Topic/subtopic dimension = 100
#   User dim = 300
#   N_Layers in GRU = 1
#   Gru output vector size = 400
#   Attention Hidden dim = 200
#   Head num = 4
#   Head dim = 100


# Import libraries
import pandas as pd
import random
import torch as th
import pickle as pkl

# Import Scripts
from Utils import ValidateModel
from Data_loaders import load_batch, User_vocab,Category_vocab,Subcategory_vocab , User_vali, UserData 

# Load Model
from LSTUR import LSTUR_con
from torch import nn,optim
device = "cuda"
max_history_length = 50


LSTUR_con_module = LSTUR_con(
    seq_len = max_history_length,
    user_dim=100,
    user_size=User_vocab.__len__(),
    topic_size=Category_vocab.__len__(),
    topic_dim=100,
    subtopic_size=Subcategory_vocab.__len__(),
    subtopic_dim=100,
    word_dim=300,
    device=device
)


#BatchSize = 400
#batches = 384 
#epochs = 10
#vali_batches = 177

BatchSize = 300
batches = 2 
epochs = 1
vali_batches = 2


model = LSTUR_con_module.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
Softmax = nn.Softmax(dim=1)


AUC = []
MRR = []
losses = []
loss_vali = []

#%%
for epoch in range(epochs):

    optimizer.zero_grad()
    BatchLoader = load_batch(UserData, batch_size=BatchSize,train = True, device=device, shuffle=True)

    for _ in range(batches):
        User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader.__next__()

        output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)

        loss = loss_fn(output, Clicked)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        losses.append(loss.item())

        print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
        #print(loss)
        
    with th.no_grad():

        BatchLoader_vali = load_batch(User_vali, batch_size=BatchSize,train = False, device=device, shuffle=False)

        for _ in range(vali_batches):
            User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader_vali.__next__()

            output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)
            pred = Softmax(output)

            loss = loss_fn(output, Clicked)
            loss_vali.append(loss.item())

            AUC_score = ValidateModel.ROC_AUC(Clicked.detach().cpu(), pred.detach().cpu())
            MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach(), pred.detach())

            AUC.append(AUC_score)
            MRR.append(MRR_score)

        # print("Validation")
        # Validation = ValidateModel(data_loader = load_batch, data = User_vali, batch_size=BatchSize, metrics = ['MRR','ROC_AUC'], device=device,train=False)
        # AUC_score, MRR_score, loss_vali_score = Validation.get_metrics(model, batches=vali_batches)

        # AUC.append(AUC_score)
        # MRR.append(MRR_score)
        # loss_vali.append(loss_vali_score)


    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(f"AUC: {AUC_score}. MRR: {MRR_score}. Loss: {loss}.")


    with open('TrainingLog.pkl', 'wb') as f:
        pkl.dump([AUC,MRR,losses,loss_vali], f)

# %%
