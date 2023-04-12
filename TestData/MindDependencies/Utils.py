import random
import re
import torch as th
from tqdm import tqdm 
from .Metrics import cal_metric

def loss_fn_vali(Scores,labels):

    loss = -th.log(th.exp(Scores[labels == 1].sum())/th.exp(Scores).sum())

    return loss

def word_tokenize(sent):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)
    
def get_mind_data_set(type):
    """Get MIND dataset address

    Args:
        type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert type in ["large", "small", "demo"]

    if type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsmall_utils.zip",
        )

    elif type == "demo":
        return (
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )
def batch_to_tensor(batch):
    user_id = th.from_numpy(batch['user_index_batch'])
    history_title = th.from_numpy(batch['clicked_title_batch'])
    impressions_title = th.from_numpy(batch['candidate_title_batch'])
    labels = th.from_numpy(batch['labels'])

    return user_id, history_title, impressions_title, labels


def run_news(model,news_filename,iterator,device):
    news_indexes = []
    news_vecs = []

    for batch in tqdm(iterator.load_news_from_file(news_filename)):
        candidate_title_batch = th.from_numpy(batch['candidate_title_batch']).to(device)
        news_vec = model.NewsEncoder(candidate_title_batch)
        news_index = batch['news_index_batch']
        news_indexes.extend(news_index)
        news_vecs.extend(news_vec)
    
    return dict(zip(news_indexes, news_vecs))

def run_user(model,news_filename,behaviors_file,iterator,device):
    user_indexes = []
    user_vecs = []

    for batch in tqdm(iterator.load_user_from_file(news_filename,behaviors_file)):
        user_id = th.from_numpy(batch['user_index_batch']).to(device)
        history_title = th.from_numpy(batch['clicked_title_batch']).to(device)
        user_vec = model.UserEncoder(user_id,history_title)
        user_index = batch['impr_index_batch']

        user_indexes.extend(user_index)
        user_vecs.extend(user_vec)

    return dict(zip(user_indexes, user_vecs))

def run_fast_eval(model, news_filename, behaviors_file, iterator, device):
    model.eval()
    with th.no_grad():
        News_vecs = run_news(model, news_filename,iterator, device)
        User_vecs = run_user(model, news_filename, behaviors_file, iterator, device)

    group_impr_indexes = []
    group_labels = []
    group_scores = []
    for impr_index,news_index,user_index,label in tqdm(iterator.load_impression_from_file(behaviors_file)):
        user_vec = User_vecs[impr_index]
        impressions_vecs = th.vstack([News_vecs[news] for news in news_index])
        Scores = user_vec @ impressions_vecs.T


        group_impr_indexes.append(impr_index)
        group_labels.append(label)
        group_scores.append(Scores.cpu().numpy())

    return group_impr_indexes, group_labels, group_scores

def validate_model(model,news_filename,behaviors_file,iterator,device,metrics):
    _, group_labels, group_scores = run_fast_eval(model, news_filename, behaviors_file, iterator, device)

    results = cal_metric(group_labels, group_scores,metrics)
    return results


