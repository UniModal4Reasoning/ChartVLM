# ChartVLM Evaluation Tools for Chart Data
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Bo Zhang, Renqiu Xia, Hancheng Ye
# All Rights Reserved 2024-2025.

import sys
sys.path.append('./ChartVLM')
from PIL import Image

from adapter.model_adapter import infer_adapter
from base_decoder.model_base_decoder import infer_base_decoder
from auxiliary_decoder.model_auxiliary_decoder import infer_auxiliary_decoder
from tools.csv2triplet import csv2triples



def infer_ChartVLM(image_path, text, model = '${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/'):

    image = Image.open(image_path)
    num = infer_adapter(text, model)

    if num == 0:   #csv
        output = infer_base_decoder(image, model, max_token=1280, title_type=False)

    if num in [1, 4]: #des sum redraw
        csv = infer_base_decoder(image, model, max_token=1280, title_type=False)
        trip = csv2triples(csv, separator='\\t', delimiter='\\n')
        title_type = infer_base_decoder(image, model, max_token=100, title_type=True)
        inputs = '<data> '+ ','.join(trip) + ' ' + title_type
        output = infer_auxiliary_decoder(text, inputs, max_token=512, model_path=model)
        output = output.split('Response:\n')[-1]


    if num == 2 : #type
        output = infer_base_decoder(image, model, max_token=100, title_type=True)
        output = output.split('<type> ')[-1]

    if num == 3: #title
        output = infer_base_decoder(image, model, max_token=100, title_type=True)
        output = output.split('<type>')[0]
        output = output.split('<title>')[-1]


    if num == 5: #QA
        csv = infer_base_decoder(image, model, max_token=1280, title_type=False)
        trip = csv2triples(csv, separator='\\t', delimiter='\\n')
        title_type = infer_base_decoder(image, model, max_token=100, title_type=True)

        inputs = '<data> '+ ','.join(trip) + ' ' + title_type.split('<type>')[-1] + ' <question> '+ text
        ins = '''
        Given the following triplet data (marked by <data>) with the title (marked by <title>) and the question related to the data (marked by <question>), give the answer with no output of hints, explanations or notes.
        '''
        output = infer_auxiliary_decoder(ins, inputs, max_token=100, model_path=model)
        output = output.split('Response:\n')[-1]

    if num > 5:
        output = 'Sorry, I can not deal with this task.'


    return output
