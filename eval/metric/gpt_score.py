from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
import csv
from tqdm import tqdm
import time
import os
import random
import json
import openai
import fire
import re


def eval_gpt_score(content, criterion, key):
    os.environ["https_proxy"] = "58.34.83.134:31280"
    openai.api_base = 'https://api.openai.com/v1'
    openai.api_key = key

    messages = [
                {
                    "role": "user",
                    "content": criterion,
                },
                {
                    "role": "user",
                    "content":  content
                }
    ]
    
    completion = openai.ChatCompletion.create(
        model = 'gpt-4-1106-preview', 
        messages = messages
    )
    data_gen = completion.choices[0].message['content']
    score = re.findall('[0-5]', data_gen)[0]
    return int(score)