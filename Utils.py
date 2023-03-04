
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

import torch as th
import numpy as np


class ValidateModel:

    def __init__(self, data_loader, data, batch_size, metrics, device = 'cpu',train=False):
        self.data_loader = data_loader(data, batch_size, train = train, device = device, shuffle = False)
        self.metrics = metrics
        self.device = device
        self.Softmax = th.nn.Softmax(dim=1)

    def mean_reciprocal_rank(self, y_true, y_score):
        order = th.argsort(y_score)[::-1]
        y_true = th.take(y_true, order)
        rr_score = y_true / (th.arange(len(y_true)) + 1)
        return rr_score.sum()/ y_true.sum()
    
    def ROC_AUC(self, y_true, y_score):
        return roc_auc_score(y_true, y_score, multi_class='ovo', labels = np.arange(0,295))
    
    def get_metrics(self,model,batches):
        
        auc_score = 0
        with th.no_grad():
            for i in range(batches):
                User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = self.data_loader.__next__()

                output = model(User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor)
                pred = self.Softmax(output)
                
                auc_score += self.ROC_AUC(Clicked.detach().cpu(),pred.detach().cpu())
                print(self.ROC_AUC(Clicked.detach().cpu(),pred.detach().cpu()))

        return auc_score/batches
