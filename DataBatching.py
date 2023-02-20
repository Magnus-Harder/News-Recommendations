# Define Vocabulary for users and topics
import random
from torchtext import vocab
from torchtext.data.utils import get_tokenizer
import torch as th
from LSTUR import GloVe

tokenizer = get_tokenizer('basic_english')


# Define Datapoint to tensor
def Datapoint_to_Encodings_train(User,News_vocab,User_vocab):

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
def Datapoint_to_Encodings_predict(User,News_vocab,User_vocab,Impression_len):

    History = News_vocab.lookup_indices(User.history.split(" "))
    User_en = User_vocab.__getitem__(User.user_id)
    Impressions = User.impressions.split(" ")
    Impressions,Clicked = map(list, zip(*[Impression.split("-") for Impression in Impressions]))
    
    

    # Convert to tensor
    Impressions = News_vocab.lookup_indices(Impressions)
    History, User_en, Impressions, Clicked = map(th.tensor, [History, User_en, Impressions, Clicked])

    Impressions_temp = -1 * th.ones(Impression_len)
    Clicked_temp = th.zeros(Impression_len)

    Impressions_temp[:len(Impressions)] = Impressions
    Clicked_temp[:len(Clicked)] = Clicked

    return History, User_en, Impressions_temp, Clicked_temp




# Pack Title
def pack_Title(title,max_length):

    src_len, _ = title.size()

    title_reformated = th.zeros(max_length,300)

    title_reformated[:src_len,:] = title

    return title_reformated, src_len


# Get Numeric Artikles representation
def get_Article_Encodings(Artikle,Category_vocab,Subcategory_vocab,max_title_len):


    title = GloVe.get_vecs_by_tokens(tokenizer(Artikle['title']))
    
    #Abstract = [tokenizer(abstract) for abstract in Artikle['abstract']]
    Category = Category_vocab.__getitem__(Artikle['category'])
    Subcategory = Subcategory_vocab.__getitem__(Artikle['subcategory'])

    title, title_len = pack_Title(title,max_title_len)

    Category, Subcategory, title_len = map(th.tensor, [Category, Subcategory, title_len])

    

    return Category, Subcategory, title, title_len

# Store all News in Dictionary for faster access
def news_dict(News,News_vocab,Category_vocab,Subcategory_vocab,max_title_len):
    News_tensors = {}

    for i in range(len(News)):
        News_tensors[News_vocab.__getitem__(News['news_id'][i])] = get_Article_Encodings(News.loc[i],Category_vocab,Subcategory_vocab,max_title_len)

    return News_tensors

# Get Numeric User representation
def Datapoint_to_tensor(User):

    History, User_en, Impressions, Clicked = Datapoint_to_Encodings(User)

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
def load_batch(User, batch_size, device='cpu'):
    
        # User = User.sample(frac=1).reset_index(drop=True)
    
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
                User_en_, Category_, Subcategory_, History_tensor_, history_len_, Category_Impressions_, Subcategory_Impressions_, Impressions_tensor_, Impressions_len_, Clicked_ = Datapoint_to_tensor(User_batch.iloc[i])
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

