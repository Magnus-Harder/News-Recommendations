
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

import torch as th
import numpy as np




def get_metrics(y_true,y_pred,metric):
    if metric == 'auc':
        return roc_auc_score(y_true,y_pred)
    

    elif metric == 'logloss':
        return log_loss(y_true,y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_true,y_pred)
    elif metric == 'accuracy':
        return accuracy_score(y_true,y_pred)
    elif metric == 'f1':
        return f1_score(y_true,y_pred)
    else:
        raise ValueError('Unknown metric: {}'.format(metric))
    

def mrr(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)
     