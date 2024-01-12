import pandas as pd
import pickle
import openai
import time
import argparse
from tqdm import trange
import json
from utils import *


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--start', type=int, default=0,
                    help="Starting index")
parser.add_argument('--end', type=int, default=12500,
                    help="Ending index")
parser.add_argument('--i', type=int, default=12500,
                    help="Ending index")

args = parser.parse_args()

start = int(args.start)
end = int(args.end)
oai_i = int(args.i)

open_ai_ids = pd.read_excel("data/GPT4-turbo-endpoints.xlsx")

openai.api_type = "azure"
openai.api_base = open_ai_ids.iloc[oai_i]['Base']
openai.api_version = open_ai_ids.iloc[oai_i]['Version']
openai.api_key = open_ai_ids.iloc[oai_i]['Key']


def gpt4(query, counter=0):
    try:
        messages = [
                       {"role": "system", "content": "You are a helpful AI assistant."},
                       {"role": "user", "content": query}
                   ]
        
        response = openai.ChatCompletion.create(
            engine="gpt-4-turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages=messages,
            temperature=0
        )
        return response['choices'][0]['message']['content'], response['usage']
    
    except:
        if counter < 15:
            time.sleep(5)
            return gpt4(query, counter + 1)
        else:
            #print("exception at:" + str(i))
            return "", {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0
}
        


knn = read_pickle('embeddings/knn.pkl')
c = pd.read_csv("data/competition.csv")
m = pd.read_csv("data/MATH.csv")


def get_prompt_for_strategy(n):
    
    # for nth competition question get prompt with 5 nn
    
    queID = c.iloc[n]['queId']
    nn = knn[queID]
    
    final_prompt = """Instruction: 
Given the math problem. Generate the right strategy which will help me solve the problem correctly. (Don't execute the strategy).

[Examples]
"""
    
    for i in nn:
        
        p, s = m.iloc[i]['problem'], m.iloc[i]['strategy']
        final_prompt += f"Question: {p}\n\nStrategy: {s}\n\n#############\n\n"
        
    final_prompt += "[End of Examples]\n\n"
    final_prompt += f"Question: {c.iloc[n]['problem']}\n\nStrategy:"
    
    return final_prompt


def get_prompt_for_execution(q, strategy):
    
    final_prompt = f"""Given the question and Strategy. Execute the Strategy on the question.\n\n{q} \n\n Strategy:{strategy}\n\n
Give your final answer in new line as Final_answer: your answer(only value NO units)
    """
    return final_prompt


Save_list = []

for i in trange(start, end):
    
    # Get strategy
    strategy_prompt = get_prompt_for_strategy(i)
    strategy, strategy_usage = gpt4(strategy_prompt)
    
    # Execute the strategy
    execution_prompt = get_prompt_for_execution(c.iloc[i]['problem'], strategy)
    res, execution_usage = gpt4(execution_prompt)
    
    
    Save_list.append({
        "index": str(i),
        'dataset_version': c.iloc[i]['dataset_version'], 
        'queId': c.iloc[i]['queId'], 
        'difficulty': str(c.iloc[i]['difficulty']), 
        'qtype': c.iloc[i]['qtype'], 
        'problem': c.iloc[i]['problem'],
        'knowledge_point_routes': c.iloc[i]['knowledge_point_routes'], 
        'strategy': strategy,
        'strategy_usage': strategy_usage,
        'pred': res,
        'pred_usage': execution_usage
    })
    
    if i%10==0:
        write_json(f"./Competition/Autoplan_competition_{start}_{end}.json", Save_list)
    
    
write_json(f"./Competition/Autoplan_competition_{start}_{end}.json", Save_list)


"""

python final_run.py --start 0 --end 593 --i 0
python final_run.py --start 593 --end 1186 --i 1
python final_run.py --start 1186 --end 1779 --i 2
python final_run.py --start 1779 --end 2372 --i 3
python final_run.py --start 2372 --end 2965 --i 4
python final_run.py --start 2965 --end 3558 --i 5
python final_run.py --start 3558 --end 4151 --i 6
python final_run.py --start 4151 --end 4744 --i 7
python final_run.py --start 4744 --end 5337 --i 8
python final_run.py --start 5337 --end 5927 --i 9

"""