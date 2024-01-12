import json
from utils import *
        
def final_extract(text):
    v = text.split("\n")[-1].replace("Final_answer", "").replace(":", "").strip()
    
    if v=="":
        return "0"
    if v[-1]==".":
        v = v[:-1]
    return v.replace("- ", "").split("=")[-1]


## concating all the completions.
from utils import *
import os

ll = []
for path in os.listdir("Competition/"):
    if path.endswith(".json"):
        print(path)
        l = read_json(f"Competition/{path}")
        ll.extend(l)
        
write_json("final_Competition_autoplan.json")




## reading final_completion

l = read_json("final_Competition_autoplan.json")

dic = {}
for i in l:
    dic[i['queId']] = final_extract(i['pred'])
    
write_json("TAL_SAQ6K_EN_prediction.json")