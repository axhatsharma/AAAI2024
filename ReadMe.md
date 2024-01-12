# AAAI2024 Track-2

## Requirements :
- Python 3.7+
- All other requirements are stored in requirements.txt. Create a new conda environment, and do
  ``` bash
  $ pip install -r requirements.txt
  ```
  
## Model used :
- Azure openai "GPT-4-turbo" version "2023-07-01-preview"
- "text-embedding-ada-002" for embeddings

## Steps :
1. Firstly, we ran the AutoPlan (https://arxiv.org/abs/2305.15064) on MATH dataset to get the strategies to solve each question.
2. Ran ada-embeddings on MATH questions and Competition questions, and got 5 nearest neighbors in MATH for each question in Competition dataset using K-nearest neighbors algorithm.
3. Using 5 nearest neighbors, we did fewshot prompting and derived strategy for each question in Competition dataset, and finally executed the derived strategy to get the final answer using GPT-4.
4. Finally, did some post processing to extract final answers from gpt-4 responses.

- Step 1:
  -   We had created 15 azure openai "gpt-4-turbo" models to run in parallel.
  -   This will generate some json files in strategies folder. Later we are going to combine all files into a single json file.

  ``` bash
  python Autoplan_to_get_strategies_for_MATH.py --start 0 --end 12496 --i 0   # if you have only 1 gpt-4 model available.
  ```
- Step 2:
  - Fill in the api_key and other details in "Run_knn.py" and then run :
         
  ``` bash
  python Run_knn.py
  ```
Step 3:
        python final_run.py
        
Step 4: 
        python submit_json.py
