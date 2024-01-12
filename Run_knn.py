import os
import time
import openai
from tqdm import tqdm, trange
import pandas as pd
import torch
import pickle
from utils import *

openai.api_type = "azure"
openai.api_base = "https://cdb-aia-guil-eastus2-gpt-01.openai.azure.com/"
openai.api_version = "2022-12-01"
KEY = "8f8e8672e1194b6f937c6ced6d1dfceb"
openai.api_key = KEY #os.getenv("OPENAI_API_KEY")


def ada(text):
    response = openai.Embedding.create(
                                        engine="text-embedding-ada-002",
                                        input=text,
                                       )
    
    return response


def get_embeddings(text):
    try:
        embed = ada(text)['data'][0]['embedding']
    except:
        print('sleeping..')
        time.sleep(2)
        return get_embeddings(text)
    return embed



## Running for MATH.csv 
df = pd.read_csv("data/MATH.csv")

Embeddings = []
for i in trange(len(df)):
    
    embed = get_embeddings(df.iloc[i]['problem'])
    Embeddings.append(embed)
    #Embeddings.append(embed)
    
write_pickle("embeddings/MATH_embeddings.pkl", Embeddings)
    


## Running for competition.csv
df = pd.read_csv("data/competition.csv")

Embeddings = []
for i in trange(len(df)):
    
    embed = get_embeddings(df.iloc[i]['problem'])
    Embeddings.append({'queId': df.iloc[i]['queId'], "embed": embed})
    #Embeddings.append(embed)
    
write_pickle("embeddings/Competition_embeddings.pkl", Embeddings)
    
    
################################

MATH = read_pickle("embeddings/MATH.pkl")
Competition = read_pickle("embeddings/Competition.pkl")

MATH = torch.tensor(MATH)

Competition_tensor = torch.tensor([i['embed'] for i in Competition])
euc_dist = torch.cdist(Competition_tensor, MATH, p=2)

knn = {}
for i in trange(5927):   # competition dataset length.
    nn = [j[0] for j in sorted(list(enumerate(euc_dist[i])), key= lambda x: x[1])[:5]]
    knn[Competition[i]['queId']] = nn
    
write_pickle("embeddings/knn.pkl", knn)


##########################################


## Extra step...
## keeping strategies in MATH.csv    
import pandas as pd
import os
from tqdm import trange
import json
from utils import *

math = pd.read_csv("data/MATH.csv")


ll = []
for path in os.listdir("strategies/"):
    if path.endswith(".json"):
        l = read_json(f"strategies/{path}")
        ll.extend(l)
        
        
strategies = []
for i in trange(len(math)):
    q = math.iloc[i]['problem']
    
    for j in ll:
        if q == j['question']:
            strategies.append(j['strategy'])
            
math['strategy'] = strategies
math.to_csv("data/MATH.csv")