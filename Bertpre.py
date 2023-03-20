##############################
# Description: Script to Define Batch Loader functions

# Import libraries
import pandas as pd
import random
from tqdm import tqdm

# Define Vocabulary for users and topics
from torchtext import vocab
from torchtext.data.utils import get_tokenizer
import torch as th

# Import libraries
import torch as th
import pandas as pd
import random
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, AdamW
from tqdm import tqdm

# Import BertTokenizer
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Import BertModel
Bert = BertModel.from_pretrained('bert-base-uncased')
dim_bert = 768
#################################
#: Load Data

# Load News
News = pd.read_csv('MINDsmall_train/news.tsv', sep='\t', header=None)
News.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

News_vali = pd.read_csv('MINDsmall_dev/news.tsv', sep='\t', header=None)
News_vali.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

News_con = pd.concat([News, News_vali], ignore_index=True)


# Load User
UserData = pd.read_csv('MINDsmall_train/behaviors.tsv', sep='\t', header=None)
UserData.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

User_vali = pd.read_csv('MINDsmall_dev/behaviors.tsv', sep='\t', header=None)
User_vali.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']


UserData = UserData.dropna()
User_vali = User_vali.dropna()

topic_size = News['category'].nunique()
subtopic_size = News['subcategory'].nunique()

max_title_length = max([len(Tokenizer(title)['input_ids']) for title in News['title']])
print("Max title length: ", max_title_length)


max_history_length = max([len(history.split(" ")) for history in UserData['history']])
max_history_length = 50 # Overwrite

impressions_length = max([len(impressions.split(" ")) for impressions in User_vali['impressions']])


print(f"Data contains {topic_size} topics and {subtopic_size} subtopics")


#################################
#: Define Vocabulary and News Dictionary

User_vocab = vocab.build_vocab_from_iterator([[id] for id in UserData['user_id']], specials=['<unk>'])
User_vocab.set_default_index(User_vocab['<unk>'])
News_vocab = vocab.build_vocab_from_iterator([[id] for id in  News_con['news_id']], specials=['<unk>'])
News_vocab.set_default_index(News_vocab['<unk>'])
Category_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['category']], specials=['<unk>'])
Category_vocab.set_default_index(Category_vocab['<unk>'])
Subcategory_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['subcategory']], specials=['<unk>'])
Subcategory_vocab.set_default_index(Subcategory_vocab['<unk>'])


# Create News_dict with news_id as key and Category and Subcategory and title as value

title_dict_train = {}
title_dict_train_mask = {}
title_dict_vali_mask = {}

title_dict_vali = {}
Category_dict_train = {}
Category_dict_vali = {}


title_token_padded = Tokenizer(News.title.values.tolist(),return_tensors='pt',max_length=max_title_length, truncation=True, padding='max_length')
title_token_padded_vali = Tokenizer(News_vali.title.values.tolist(),return_tensors='pt',max_length=max_title_length, truncation=True, padding='max_length')


# Create News_dict with news_id as key and Category and Subcategory and title as value
for idx, (id, Category, SubCategory) in tqdm(enumerate(zip(News.news_id, News.category, News.subcategory))):
    with th.no_grad():
    
        bertout = Bert(
            input_ids=title_token_padded['input_ids'][idx].unsqueeze(0),
            attention_mask=title_token_padded['attention_mask'][idx].unsqueeze(0),
            token_type_ids=title_token_padded['token_type_ids'][idx].unsqueeze(0)
        )

    title_dict_train[News_vocab.lookup_indices([id])[0]] = bertout.pooler_output[0]

    Category_dict_train[News_vocab.lookup_indices([id])[0]] = (Category_vocab.__getitem__(Category), Subcategory_vocab.__getitem__(SubCategory))


for idx, (id, Category, SubCategory) in enumerate(zip(News_vali.news_id, News_vali.category, News_vali.subcategory)):
    with th.no_grad():

        bertout = Bert(
            input_ids=title_token_padded_vali['input_ids'][idx].unsqueeze(0),
            attention_mask=title_token_padded_vali['attention_mask'][idx].unsqueeze(0),
            token_type_ids=title_token_padded_vali['token_type_ids'][idx].unsqueeze(0)
        )

    title_dict_vali[News_vocab.lookup_indices([id])[0]] = bertout.pooler_output[0]

    Category_dict_vali[News_vocab.lookup_indices([id])[0]] = (Category_vocab.__getitem__(Category), Subcategory_vocab.__getitem__(SubCategory))


# Define Datapoint to tensor
def Datapoint_to_Encodings(User):

    History = News_vocab.lookup_indices(User.history.split(" "))
    User_en = User_vocab.__getitem__(User.user_id)
    Impressions = User.impressions.split(" ")
    Impressions,Clicked = map(list, zip(*[Impression.split("-") for Impression in Impressions]))
    
    Positive, Negative = [],[]
    for idx, click in enumerate(Clicked):
        if click == "1":
            Positive.append(Impressions[idx])
        else:
            Negative.append(Impressions[idx])

    Impressions = [Positive[0]]
    random.seed(0)
   

    if len(Negative) > 3:
        for _ in random.sample(Negative,4):
            Impressions.append(_)
    else:
        for _ in range(4):
            Impressions.append(random.choice(Negative))

    Clicked = [1,0,0,0,0]

    # Shuffle
    shuffled_index = [0,1,2,3,4]
    random.shuffle(shuffled_index)


    Impressions = [Impressions[i] for i in shuffled_index]
    Clicked = [Clicked[i] for i in shuffled_index]


    # Convert to tensor
    Impressions = News_vocab.lookup_indices(Impressions)
    History, User_en, Impressions, Clicked = map(th.tensor, [History, User_en, Impressions, Clicked])

    return History, User_en, Impressions, Clicked

# Define Datapoint to tensor
def Datapoint_to_Encodings_vali(User):

    History = News_vocab.lookup_indices(User.history.split(" "))
    User_en = User_vocab.__getitem__(User.user_id)
    Impressions = User.impressions.split(" ")
    Impressions,Clicked = map(list, zip(*[Impression.split("-") for Impression in Impressions]))
    

    # Convert to tensor
    Impressions = News_vocab.lookup_indices(Impressions)
    Clicked = [int(click) for click in Clicked]

    History, User_en, Impressions, Clicked = map(th.tensor, [History, User_en, Impressions, Clicked])

    return History, User_en, Impressions, Clicked



# Get Numeric Artikles representation
def get_Article_Encodings(Artikle,train=True):

    Artikle = Artikle.item()

    if train:
        title = title_dict_train[Artikle]
        Category, Subcategory = Category_dict_train[Artikle]

    else:
        title = title_dict_vali[Artikle]
        Category, Subcategory = Category_dict_vali[Artikle]


    Category, Subcategory = map(th.tensor, [Category, Subcategory])

    return Category, Subcategory, title


# Get Numeric User representation
def Datapoint_to_tensor(User,train=True):

    if train:
        History, User_en, Impressions, Clicked = Datapoint_to_Encodings(User)
        max_impressions_length = 5
        if random.random() < 0.4:
            User_en = th.tensor(0) # Mask user 
    else:
        max_impressions_length = impressions_length
        History, User_en, Impressions, Clicked = Datapoint_to_Encodings_vali(User)


    # Create Tensor for User
    History_tensor = th.zeros(max_history_length,dim_bert)
    Category = th.zeros(max_history_length)
    Subcategory = th.zeros(max_history_length)

    # Get history length of user
    history_len = min(len(History),max_history_length)

    for idx,article in enumerate(History[-history_len:]):
        Category[idx], Subcategory[idx], History_tensor[idx] = get_Article_Encodings(article,train=train)

    # Create Tensor for Impressions
    Impressions_tensor = th.zeros(max_impressions_length,dim_bert)
    Category_Impressions = th.zeros(max_impressions_length)
    Subcategory_Impressions = th.zeros(max_impressions_length)
    Impressions_len = len(Impressions)

    history_len, Impressions_len = map(th.tensor, [history_len, Impressions_len])


    for idx,article in enumerate(Impressions):
        Category_Impressions[idx], Subcategory_Impressions[idx], Impressions_tensor[idx] = get_Article_Encodings(article,train=train)
    
    Clicked = Clicked.argmax()

    return User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked


# Def load batch
def load_batch(User, batch_size, device='cpu',train=True, shuffle=False):

    if shuffle:
        User = User.sample(frac=1).reset_index(drop=True)

    for i in range(0, len(User), batch_size):

        User_batch = User[i:i+batch_size]

        User_en = []
        Category = []
        Subcategory = []
        History_tensor = []
        history_len = []
        Category_Impressions = []
        Subcategory_Impressions = []
        Impressions_tensor = []
        Impressions_len = []
        Clicked = []

        for i in range(len(User_batch)):
            User_en_, Category_, Subcategory_, History_tensor_, history_len_, Category_Impressions_, Subcategory_Impressions_, Impressions_tensor_, Impressions_len_, Clicked_ = Datapoint_to_tensor(User_batch.iloc[i],train=train)
            User_en.append(User_en_)
            Category.append(Category_)
            Subcategory.append(Subcategory_)
            History_tensor.append(History_tensor_)
            history_len.append(history_len_)
            Category_Impressions.append(Category_Impressions_)
            Subcategory_Impressions.append(Subcategory_Impressions_)
            Impressions_tensor.append(Impressions_tensor_)
            Impressions_len.append(Impressions_len_)
            Clicked.append(Clicked_)
        
        User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked = map(th.stack, [User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked])
        
        # Map tensors to long

        User_en, Category, Subcategory, history_len, Category_Impressions, Subcategory_Impressions, Impressions_len, Clicked = map(lambda x: x.long(), [User_en, Category, Subcategory, history_len, Category_Impressions, Subcategory_Impressions, Impressions_len, Clicked])

        yield User_en.to(device), Category.to(device), Subcategory.to(device), History_tensor.to(device), history_len.to(device), Category_Impressions.to(device), Subcategory_Impressions.to(device), Impressions_tensor.to(device), Impressions_len.to(device), Clicked.to(device)

