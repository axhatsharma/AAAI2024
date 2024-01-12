import pickle
import json


def read_pickle(path):
    with open(path, 'rb') as f:
        ll = pickle.load(f)
        
    return ll

def write_pickle(path, ll):
    with open(path, 'wb') as f:
        pickle.dump(ll, f)
        

def read_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return data

def write_json(path, s):
    with open(path, 'w') as f:
        json.dump(s, f)