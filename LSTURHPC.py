#%%
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
    user_dim=300,
    user_size=User_vocab.__len__(),
    topic_size=Category_vocab.__len__(),
    topic_dim=100,
    subtopic_size=Subcategory_vocab.__len__(),
    subtopic_dim=100,
    word_dim=300,
    device=device
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(LSTUR_con_module.parameters(), lr=0.001)

model = LSTUR_con_module.to(device)

BatchSize = 100
batches = 1537  
epochs = 10
vali_batches = 709

# BatchSize = 100
# batches = 10  
# epochs = 2
# vali_batches = 2


model = LSTUR_con_module.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
Softmax = nn.Softmax(dim=1)


AUC = []
MRR = []
losses = []
loss_vali = []

#%%
for epoch in range(epochs):

    BatchLoader = load_batch(UserData, batch_size=BatchSize,train = True, device=device, shuffle=True)

    for _ in range(batches):
        User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader.__next__()

        optimizer.zero_grad()

        output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)

        loss = loss_fn(output, Clicked)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        #print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
        #print(loss)
        
    with th.no_grad():
        print("Validation")
        Validation = ValidateModel(data_loader = load_batch, data = User_vali, batch_size=BatchSize, metrics = ['MRR','ROC_AUC'], device=device,train=False)
        AUC_score, MRR_score, loss_vali_score = Validation.get_metrics(model, batches=vali_batches)

        AUC.append(AUC_score)
        MRR.append(MRR_score)
        loss_vali.append(loss_vali_score)


    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(f"AUC: {AUC_score}. MRR: {MRR_score}. Loss: {loss_vali_score}.")


    with open('TrainingLog.pkl', 'wb') as f:
        pkl.dump([AUC,MRR,losses,loss_vali], f)

# %%
