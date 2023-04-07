#%%
import torch as th


def loss_fn(Scores):
    return -th.log(th.exp(Scores[:,:,0])/th.exp(Scores).sum(dim=2)).sum(dim=1).mean()

# %%
