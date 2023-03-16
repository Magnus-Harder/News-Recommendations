########################################################################################
#'Script used to train LSTUR model on HPC in order to recreate results from LSTUR-paper.
#'Hyperparameters used in paper:
#'   Word embedding dim = 300
#'   Filters in CNN = 300
#'   Windowsize CNN = 3
#'   Dropout rate = 0.2
#'   Masking probability = 0.5
#'   Adam optimizer learning rate = 0.01
#'   Batch size = 400
#'   Negative samples pr positive sample = 4
#'   Epochs = 10
#'   History length = 50
#'   Topic/subtopic dimension = 100
#'   User dim = 300
#'   N_Layers in GRU = 1
#'   Gru output vector size = 400
#'   Attention Hidden dim = 200
#'   Head num = 4
#'   Head dim = 100
###########################################################################################

# Import libraries
import torch as th
import pickle as pkl

# Import Scripts
from Utils import ValidateModel
from Data_loaders_Demo import load_batch, User_vocab,Category_vocab,Subcategory_vocab , User_vali, UserData 

# Load Model
from LSTURini import LSTURini
from torch import nn,optim
device = "cuda" if th.cuda.is_available() else "mps"
max_history_length = 50

# Set Model Architecture
LSTUR_con_module = LSTURini(
    seq_len = max_history_length,
    user_dim=400,
    user_size=User_vocab.__len__(),
    topic_size=Category_vocab.__len__(),
    topic_dim=50,
    subtopic_size=Subcategory_vocab.__len__(),
    subtopic_dim=50,
    word_dim=300,
    device=device
)

# Set hyperparameters for data loading and training
BatchSize = 32
batches =  int(len(UserData)/BatchSize) 
epochs = 5
vali_batches = int(len(User_vali))

# Initialize model on device, loss function and optimizer
model = LSTUR_con_module.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
Softmax = nn.Softmax(dim=1)


# Pre Training Validation step
model.train(False)
with th.no_grad():
    
    # Initialize variables
    AUC_pre= 0
    MRR_pre= 0
    loss_pre = 0

    # Load validation data
    BatchLoader_vali = load_batch(User_vali, batch_size=1,train = False, device=device, shuffle=False)

    # Loop through validation data
    for _ in range(vali_batches):

        # Load batch
        User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader_vali.__next__()

        # Get length of impression
        idx = Impressions_len.item()

        # Get output
        output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions[:,:idx], Subcategory_Impressions[:,:idx], Impressions_tensor[:,:idx])
        pred = Softmax(output)

        # Calculate loss
        loss = loss_fn(output, Clicked)
        loss_pre += loss.item()/vali_batches

        # Calculate metrics
        AUC_score = ValidateModel.ROC_AUC(Clicked.item(), pred.detach().cpu()[0],Impressions_len.item())
        MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach().cpu(), pred.detach().cpu()[0])

        AUC_pre += AUC_score/vali_batches
        MRR_pre += MRR_score.item()/vali_batches


print(f"Pre Training AUC: {AUC_pre}, MRR: {MRR_pre}, Loss: {loss_pre}")

# Initialize lists for saving metrics
AUC = [AUC_pre]
MRR = [MRR_pre]
losses = []
loss_vali = [loss_pre]

# Training loop
for epoch in range(epochs):

    # Training step
    model.train(True)
    optimizer.zero_grad()

    # Load training data
    BatchLoader = load_batch(UserData, batch_size=BatchSize,train = True, device=device, shuffle=True)

    # Loop through training data
    for _ in range(batches):

        # Load batch
        User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader.__next__()

        # Get output
        output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)

        # Calculate loss
        loss = loss_fn(output, Clicked)
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
        BatchLoader_vali = load_batch(User_vali, batch_size=1,train = False, device=device, shuffle=False)

        # Loop through validation data
        for _ in range(vali_batches):

            # Load batch
            User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = BatchLoader_vali.__next__()

            # Get length of impression 
            idx = Impressions_len.item()

            # Get output
            output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions[:,:idx], Subcategory_Impressions[:,:idx], Impressions_tensor[:,:idx])
            pred = Softmax(output)

            # Calculate loss
            loss = loss_fn(output, Clicked)
            loss_vali_epoch += loss.item()/vali_batches

            # Calculate AUC and MRR
            AUC_score = ValidateModel.ROC_AUC(Clicked.item(), pred.detach().cpu()[0],Impressions_len.item())
            MRR_score = ValidateModel.mean_reciprocal_rank(Clicked.detach(), pred.detach()[0])

            AUC_epoch += AUC_score/vali_batches
            MRR_epoch += MRR_score.item()/vali_batches

        # Save loss, AUC and MRR
        loss_vali.append(loss_vali_epoch)
        AUC.append(AUC_epoch)
        MRR.append(MRR_epoch)


    print(f'Memory: {th.cuda.memory_reserved()/(10**9)} GB')
    print(f"AUC: {AUC_epoch}. MRR: {MRR_epoch}. Loss: {loss_vali_epoch}.")

# Saving Training Log
with open('TrainingLogDemo.pkl', 'wb') as f:
    pkl.dump([AUC,MRR,losses,loss_vali], f)
