##############################
# Description: Script to Define Batch Loader functions

# Import libraries
import pandas as pd
import random

# Define Vocabulary for users and topics
from torchtext import vocab
from torchtext.data.utils import get_tokenizer
import torch as th
from LSTUR import GloVe


# Load tsv file
News_vali = pd.read_csv('MINDsmall_dev/news.tsv', sep='\t', header=None)
News_vali.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

User_vali = pd.read_csv('MINDsmall_dev/behaviors.tsv', sep='\t', header=None)
User_vali.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']


# Load tsv file
News = pd.read_csv('MINDsmall_train/news.tsv', sep='\t', header=None)
News.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
News_vali = pd.read_csv('MINDsmall_dev/news.tsv', sep='\t', header=None)
News_vali.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

News_con = pd.concat([News, News_vali], ignore_index=True)


UserData = pd.read_csv('MINDsmall_train/behaviors.tsv', sep='\t', header=None)
UserData.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

UserData = UserData.dropna()
User_vali = User_vali.dropna()

topic_size = News['category'].nunique()
subtopic_size = News['subcategory'].nunique()

print(f"Data contains {topic_size} topics and {subtopic_size} subtopics")




tokenizer = get_tokenizer('basic_english')

User_vocab = vocab.build_vocab_from_iterator([[id] for id in UserData['user_id']], specials=['<unk>'])
User_vocab.set_default_index(User_vocab['<unk>'])
News_vocab = vocab.build_vocab_from_iterator([[id] for id in  News_con['news_id']], specials=['<unk>'])
News_vocab.set_default_index(News_vocab['<unk>'])
Category_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['category']], specials=['<unk>'])
Category_vocab.set_default_index(Category_vocab['<unk>'])
Subcategory_vocab = vocab.build_vocab_from_iterator([[Category] for Category in News['subcategory']], specials=['<unk>'])
Subcategory_vocab.set_default_index(Subcategory_vocab['<unk>'])


# Define Vocabulary for title and abstract
max_title_length = max([len(tokenizer(title)) for title in News['title']])
max_history_length = max([len(history.split(" ")) for history in UserData['history']])
max_history_length = 50 # Overwrite

impressions_length = max([len(impressions.split(" ")) for impressions in User_vali['impressions']])
#max_impressions_length = 5 # Overwrite

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

# Pack Title
def pack_Title(title,max_length):

    src_len, _ = title.size()

    title_reformated = th.zeros(max_length,300)

    title_reformated[:src_len,:] = title

    return title_reformated, src_len


# Get Numeric Artikles representation
def get_Article_Encodings(Artikle):


    title = GloVe.get_vecs_by_tokens(tokenizer(Artikle['title']))
    
    #Abstract = [tokenizer(abstract) for abstract in Artikle['abstract']]
    Category = Category_vocab.__getitem__(Artikle['category'])
    Subcategory = Subcategory_vocab.__getitem__(Artikle['subcategory'])

    title, title_len = pack_Title(title,max_title_length)

    Category, Subcategory, title_len = map(th.tensor, [Category, Subcategory, title_len])

    

    return Category, Subcategory, title, title_len

# Store all News in Dictionary for faster access
News_tensors = {}

for i in range(len(News_con)):
    News_tensors[News_vocab.__getitem__(News_con['news_id'][i])] = get_Article_Encodings(News_con.loc[i])

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


    History_tensor = th.zeros(max_history_length,max_title_length,300)
    Category = th.zeros(max_history_length)
    Subcategory = th.zeros(max_history_length)
    history_len = min(len(History),max_history_length)

    for idx,article in enumerate(History[-history_len:]):
        Category[idx], Subcategory[idx], History_tensor[idx], _ = News_tensors[article.item()]

    Impressions_tensor = th.zeros(max_impressions_length,max_title_length,300)
    Category_Impressions = th.zeros(max_impressions_length)
    Subcategory_Impressions = th.zeros(max_impressions_length)
    Impressions_len = len(Impressions)

    history_len, Impressions_len = map(th.tensor, [history_len, Impressions_len])


    for idx,article in enumerate(Impressions):
        Category_Impressions[idx], Subcategory_Impressions[idx], Impressions_tensor[idx], _ = News_tensors[article.item()]
    
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
        User_en, Category, Subcategory, history_len, Category_Impressions, Subcategory_Impressions, Impressions_len, Clicked = map(lambda x: x.long(), [User_en, Category, Subcategory, history_len, Category_Impressions, Subcategory_Impressions, Impressions_len, Clicked])
        yield User_en.to(device), Category.to(device), Subcategory.to(device), History_tensor.to(device), history_len.to(device), Category_Impressions.to(device), Subcategory_Impressions.to(device), Impressions_tensor.to(device), Impressions_len.to(device), Clicked.to(device)

        #yield User_en, Category, Subcategory, History_tensor, history_len, Category_Impressions, Subcategory_Impressions, Impressions_tensor, Impressions_len, Clicked

