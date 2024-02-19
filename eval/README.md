You can evaluate ChartVLM model on ChartX benchmark via the following steps:

```
cd eval
```

## Download benchmark ChartX
Put ChartX under the path `eval/ChartX` 

## Inference using ChartVLM 
Use ChartVLM model to perform the inference process:

```
python infer_ChartX.py \
--model_path ${PATH_TO_CHARTVLM_MODEL}$ \
--benchmark_json_path ${PATH_TO_JSON_FILE_IN_OFIICIAL_CHARTX_FOLDER}$ 
```
The output json file will be saved under `eval/infer_result`.

## Evaluation on chart-related tasks
We adopt Structuring Chart-oriented Representation Metric (SCRM) for Structrual Extraction (SE) task, GPT-acc metric for Question-Answering task, and GPT-score metric for description task, summarization task and redrawing code task.

`${PATH_TO_INFERENCE_RESULT_JSON_FILE}$` indicates the inference results saved in `eval/infer_result`.

*For Structrual Extraction (SE):
```
python eval_SE_ChartX.py \ 
--infer_result_dir ${PATH_TO_INFERENCE_RESULT_JSON_FILE}$ 
```

*For Question Answering:
```
python eval_qa_ChartX.py \
--infer_result_dir ${PATH_TO_INFERENCE_RESULT_JSON_FILE}$ \ 
--your_openai_key ${YOUR_OPENAI_KEY}$
```

*For Description:
```
python eval_des_ChartX.py \
--infer_result_dir ${PATH_TO_INFERENCE_RESULT_JSON_FILE}$ \ 
--your_openai_key ${YOUR_OPENAI_KEY}$
```

*For Summarization:
```
python eval_sum_ChartX.py \
--infer_result_dir ${PATH_TO_INFERENCE_RESULT_JSON_FILE}$ \ 
--your_openai_key ${YOUR_OPENAI_KEY}$
```

*For Redrawing code:
```
python eval_redraw_code_ChartX.py \
--infer_result_dir ${PATH_TO_INFERENCE_RESULT_JSON_FILE}$ \ 
--your_openai_key ${YOUR_OPENAI_KEY}$
```
Note that all the evaluation result `log` file will be saved under `eval/eval_result`

