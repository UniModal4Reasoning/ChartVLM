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


def eval_gpt_acc(question, answer_gt, answer_pred, key):
    os.environ["https_proxy"] = "58.34.83.134:31280"
    openai.api_base = 'https://api.openai.com/v1'
    openai.api_key = key

    examples = [
        {
            "query": "<question> What was the incremental increase in revenue from 2020 to 2021? <groundtruth answer> 5 million $ <answer> 20\n</s>",
            "answer": "False"
        },{
            "query": "<question> What percentage of government spending was allocated to infrastructure in 2020? <groundtruth answer> 10% <answer> 14-4=10\n</s>",
            "answer": "True"
        },{
            "query": "<question> What is the total production of Wind Energy in the four months from January to April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January to April 2021 is 2450 MW.",
            "answer": "True"
        },{
            "query": "<question> What is the total of manufactured goods for UK and Germany combined? <groundtruth answer> 5 <answer> Five",
            "answer": "True"
        },
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # instruction
    prefix = f"""Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions. The output should be only "True" or "False". Note that if the groundtruth answer is a numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is marked as correct when groundtruth value is 100 million."""
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """
    
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    query = f"<question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}"

    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo', 
        messages =[
            {"role": "user",
            "content": few_shot_prompt_template.format(
                query=query
                )
            }
        ]
    )
    
    data_gen = completion.choices[0].message['content']
    if 'True' in data_gen:
        acc = 1
    if 'False' in data_gen: 
        acc = 0

    return acc