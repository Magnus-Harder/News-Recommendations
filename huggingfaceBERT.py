#%%
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

print('Hello World')
# %%


test_senctence = "Title Descriping and Article for a news paper: How to Wrtie the perfekt article for a news paper"
encoded_senctence = tokenizer(test_senctence,return_tensors='pt') 
output = model(**encoded_senctence)

print(output)


# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input1 = tokenizer(text, return_tensors='pt')
output = model(**encoded_input1)
# %%
