
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

import torch as th
import numpy as np

######
# 1. 10 pad 330
# 2. 20 pad 330


# [0.02,0.22,0.023,----, x, x,x,x,x,x,...]




class ValidateModel:

    def __init__(self, data_loader, data, batch_size, metrics, device = 'cpu',train=False):
        self.data_loader = data_loader(data, batch_size, train = train, device = device, shuffle = False)
        self.metrics = metrics
        self.device = device
        self.Softmax = th.nn.Softmax(dim=1)

    # @staticmethod
    # def mean_reciprocal_rank(y_true, y_score):
    #     N,n_classes = y_score.shape
    #     order = th.topk(y_score,k=n_classes).indices
    #     rank = th.take(order,y_true) +1
    #     rr_score = 1 / rank
    #     return rr_score.sum()/ N

    @staticmethod
    def mean_reciprocal_rank(y_true, y_score):
        order = th.topk(y_score,k=len(y_score)).indices
        rank = th.take(order,y_true) +1
        rr_score = 1 / rank
        return rr_score
    
    #@staticmethod
    # def ROC_AUC(y_true, y_score,n_classes=295):
    #     return roc_auc_score(y_true, y_score, multi_class='ovo', labels = np.arange(0,n_classes))
    
    @staticmethod
    def ROC_AUC(y_true, y_score,Impression_len):
        y_true_binary = th.zeros(Impression_len)
        y_true_binary[y_true] = 1

        return roc_auc_score(y_true_binary, y_score)

    
    def get_metrics(self,model,batches):

        loss_fn = th.nn.CrossEntropyLoss()
        
        auc_score = 0
        MRR_score = 0
        loss_score = 0
        with th.no_grad():
            for i in range(batches):
                User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = self.data_loader.__next__()

                output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)
                pred = self.Softmax(output)
                

                MRR = self.mean_reciprocal_rank(Clicked,pred).item()
                AUC = self.ROC_AUC(Clicked.detach().cpu(),pred.detach().cpu())
                loss = loss_fn(output, Clicked)
                

                auc_score += AUC
                MRR_score += MRR
                loss_score += loss.item()

                #print(f"Matrics Batch: AUC: {AUC} MRR: {MRR}")


        return auc_score/batches , MRR_score/batches, loss_score/batches
