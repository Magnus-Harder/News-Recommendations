import torch as th
import torchtext
import sklearn
import numpy as np
import pandas as pd
import random
import pickle
from LSTUR import GloVe

print("Hello HPC")


with open('TestSave.pkl' , 'wb') as f:
	pickle.dump([0,0,0],f)

list_test = [1,2,3,4,5,6,7,8,9,10]
print(list_test)
