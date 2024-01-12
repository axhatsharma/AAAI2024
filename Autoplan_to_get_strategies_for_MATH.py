import os
import time
import json 
import regex
import random
import pickle 
import re
import sys
import argparse
import numpy as np
import openai
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
from tqdm import tqdm, trange
from utils import *
import logging


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
 
# Create and configure logger
logging.basicConfig(filename=f"logs/Autoplan_{start}_{end}.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)



def pretty_print(role, text, verbose=False):
    string = '------------{}-----------\n{}\n'.format(role, text)
    if verbose: print(string, end=''), sys.stdout.flush()
    return string


class Autoplan:
    def __init__(self, lm, method, backend_args, quota_args, **kwargs):
        self.lm = lm
        self.method = method  # step
        self.backend = backend_args['name']  # openai
        
        if self.backend == 'openai':
            openai.api_type = "azure"#quota_args["api_type"]
            openai.api_base = quota_args["api_base"]
            openai.api_version = quota_args["api_version"]
            openai.api_key = quota_args["api_key"]
    
        self.top_p = backend_args['top_p']
        self.temp = backend_args['temp']
        self.max_token = backend_args['max_token']
        self.presence_penalty = backend_args['presence_penalty']        
        
        self.max_iter_per_instance = quota_args['max_iter_per_instance']

        
        self.history = []
        self.strategy = None

        # openai api
        self.n_prompt_token = 0
        self.n_sample_token = 0
        self.messages = []
        
    def task_desription(self):
        return '''Solve a Math question answering task'''
    
    def score(self, instance, prediction):
        question, answer = instance['question'], instance['answer']
        query = """Question: "{}"\n
Gold Answer: "{}"\n
My Prediction: "{}"\n
Verify whether my prediction to the question is equivalent to gold answer. Respond with yes/no.""".format(question, answer, prediction)

        message = {
            'role': 'user',
            'content': query
        }
        
        while True:
            try:
                response = func_timeout(10, 
                    openai.ChatCompletion.create,
                    kwargs={
                        "engine": "gpt-4",
                        "messages": [message],
                        "temperature": 0.
                    }                
                )['choices'][0]['message']['content'].lower()
                break
            except FunctionTimedOut:
                print('OpenAI API call timeout')
                continue
        
        assert 'yes' in response or 'no' in response, (question, answer, prediction, response)
        return 'yes' in response
    
    def eval(self, instances, predictions):
        n_correct = 0
        correct_mask = []
        for idx in tqdm(range(len(instances)), desc='Evaluating'):
            result = self.score(instances[idx], predictions[idx])
            n_correct += int(result)
            correct_mask.append(result)
        
        return {'n_correct': n_correct, 'correct_mask': correct_mask}
    
    
    def price(self):
        price = PRICE[self.lm]
        return (self.n_prompt_token * price['prompt'] + self.n_sample_token * price['sample']) / 1000
    
    
    def call_openai_api(self, messages, stop, lm=None, top_p=None):
        n_try = 10
        while n_try > 0:
            try:
                #time.sleep(1)
                response =  openai.ChatCompletion.create(
                       engine="gpt-4-turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
                        messages=messages,
                        temperature=self.temp,
                        max_tokens= self.max_token,
                        presence_penalty=self.presence_penalty,
                        stop=stop,
                    )
                
                break
            except FunctionTimedOut:
                print('[LOG] OpenAI API call timeout')
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
            except Exception as e:
                print('[LOG]', e)
                time.sleep(15)
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
        return response
        
    def call_lm(self, prompt, add_response=True, stop=None, lm=None, top_p=None):
        
        self.messages.append({'role': 'user', 'content': prompt})
        response = self.call_openai_api(self.messages, stop, lm=lm, top_p=top_p)
        if add_response: self.messages.append(response['choices'][0]['message'])
        self.n_prompt_token += response['usage']['prompt_tokens']
        self.n_sample_token += response['usage']['completion_tokens']
        return response['choices'][0]['message']['content']
        
    def init(self):
        prompt = "{}\n\nTask Plan:".format(self.task_desription())
        strategy = self.call_lm(prompt)
        return strategy
    
    
    def run(self, data, strategy=None, is_test=False, verbose=False, return_history=False):
        questions = []
        answers = []
        predictions = []
        summaries = []
        flawed_actions = []
        flawed_plans = []
        history = ''

        
        for q_idx in range(len(data)):
            self.messages = []
            init_msg = self.task_desription()
            if 'direct' not in self.method and strategy is not None:
                init_msg += "\n\nTask Plan:\n{}\n".format(strategy)
            init_msg += '\n'

            question = data[q_idx]['question']
            answer = data[q_idx]['solution']
            questions.append(question)
            answers.append(answer)
            
            input_msg = init_msg + '\nQuestion: ' + question + '\n'
            solution = self.call_lm(input_msg)
            
            #history += pretty_print('Human', input_msg, verbose)
            logger.info(pretty_print('Human', input_msg, verbose))
            logger.info(solution)
            #print(solution)
            
            if not is_test:
                
                #summary_msg = 'Task finished.'

                summary_msg = ' The ground truth solution is "{}". Summarize the interaction history concisely.'.format(answer)

                #history += pretty_print('Human', summary_msg, verbose)
                logger.info(pretty_print('Human', summary_msg, verbose))
                
                summary = self.call_lm(summary_msg, top_p=0.)
                #history += pretty_print('Machine', summary, verbose)
                logger.info(pretty_print('Machine', summary, verbose))
                
                summaries.append(summary)
                
                if strategy is not None:

                    failed_action_msg = 'Identify all flawed parts of the plan (not flawed action). Only the flawed part.'
                    #history += pretty_print('Human', failed_action_msg, verbose)
                    logger.info(pretty_print('Human', failed_action_msg, verbose))
                    
                    failed_action = self.call_lm(failed_action_msg, top_p=0.)
                    #history += pretty_print('Machine', failed_action, verbose)
                    logger.info(pretty_print('Machine', failed_action, verbose))
                    
                    
                    flawed_actions.append(failed_action)

                    suggest_rev_msg = 'Suggest revision to the current flawed part of the plan. Only the flawed part.'
                    #history += pretty_print('Human', suggest_rev_msg, verbose)
                    logger.info(pretty_print('Human', suggest_rev_msg, verbose))
                    
                    suggest_rev = self.call_lm(suggest_rev_msg, stop=None, top_p=0.0)
                    #history += pretty_print('Machine', suggest_rev, verbose)
                    logger.info(pretty_print('Machine', suggest_rev, verbose))
                    
                    flawed_plans.append(suggest_rev)
                    
            #print("\n\n############\n\nID Done:", q_idx, "\n\n")
            
            # with open("summaries.json", 'w') as f:
            #     json.dump({"questions":questions, "answers":answers, "summaries": summaries, "flawed_actions": flawed_actions, "flawed_plans":flawed_plans}, f)
            

        to_return = None

        if is_test:
            to_return = predictions
        else:
            self.messages = []
            final_msg = 'Task Description:\n' + self.task_desription() + '\n\n'
            final_msg += 'Current Task Plan:\n{}\n\n'.format(strategy)
            final_msg += '=' * 10 + 'Task Experiences Begin' + '=' * 10 + '\n\n'

            
            for q_idx in range(len(data)):
                question = data[q_idx]['question']
                final_msg += 'Job {}: Answering the following question. {}\nSummary of Job {}:\n{}\n'.format(q_idx, question, q_idx, summaries[q_idx])
                if strategy is not None:
                    final_msg += 'Flaws of Plan in Job {}:\n{}\n'.format(q_idx, flawed_actions[q_idx])
                    final_msg += 'Suggested Revision of Plan from Job {}:\n{}\n'.format(q_idx, flawed_plans[q_idx])

            final_msg += '=' * 10 + 'Task Experiences End' + '=' * 10 + '\n\n'

            final_msg += 'Based on the above {} experience of the task, rewrite the current task plan. The plan should be specific to this question and instead of providing the solution give a specific approach.  \nNew Task Plan:'.format(len(data))

            #history += pretty_print('Human', final_msg, verbose)
            logger.info(pretty_print('Human', final_msg, verbose))
            
            # with open("final_msg.txt", "w") as f:
            #     f.write(final_msg)
                
            new_strategy = self.call_lm(final_msg, top_p=0.)
            
            
            #history += pretty_print('Machine', new_strategy, verbose)
            logger.info(pretty_print('Machine', new_strategy, verbose))
            
            to_return = new_strategy

        self.history.append(history)

        if return_history:
            to_return = to_return, history
            return to_return

        # if not is_test:
        #     n_correct = self.eval(data, predictions)['n_correct']

        return to_return, history
    

backend_args = {
        'name': "openai",
        'top_p': 1,
        'temp': 0,
        'max_token': 3000,
        'presence_penalty': 1.5,
    }

open_ai_ids = pd.read_excel("data/GPT4-turbo-endpoints.xlsx")

quota_args = {
        'sleep_minutes': 1,
        'max_iter_per_instance': 4,
        "engine": open_ai_ids.iloc[args.i]['Deployment'],
        "api_base": open_ai_ids.iloc[args.i]['Base'],
        "api_version": open_ai_ids.iloc[args.i]['Version'],
        "api_key":  open_ai_ids.iloc[args.i]['Key']
}


df = pd.read_csv("data/MATH.csv")
        
data = [{"question": df.iloc[i]['problem'], "solution": df.iloc[i]['solution']} for i in range(start, end)]

for batch in trange(0, len(data)):
    
    autoplan = Autoplan("gpt-4-turbo", 'step', backend_args, quota_args)
    
    strategy = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."          # Plan and solve prompting..
    
    new_strategy, history = autoplan.run([data[batch]],
                                    strategy=strategy,
                                    is_test=False,
                                    verbose=False,
                                    return_history=True)
    
    data[batch]['strategy'] = new_strategy
    
    write_json(f'./strategies/strategy_{start}_{end}.json', data)
        

        
"""

python autoplan.py --start 0 --end 830  --i 0
python autoplan.py --start 830 --end 1660  --i 1
python autoplan.py --start 1660 --end 2490  --i 2
python autoplan.py --start 2490 --end 3320  --i 3
python autoplan.py --start 3320 --end 4150  --i 4
python autoplan.py --start 4150 --end 4980  --i 5
python autoplan.py --start 4980 --end 5810  --i 6
python autoplan.py --start 5810 --end 6640  --i 7
python autoplan.py --start 6640 --end 7470  --i 8
python autoplan.py --start 7470 --end 8300  --i 9
python autoplan.py --start 8300 --end 9130  --i 10
python autoplan.py --start 9130 --end 9960  --i 11
python autoplan.py --start 9960 --end 10790  --i 12
python autoplan.py --start 10790 --end 11620  --i 13
python autoplan.py --start 11620 --end 12496  --i 14

"""