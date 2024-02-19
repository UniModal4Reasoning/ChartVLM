<div align="center">
<h1>Instrcution adapter for ChartVLM<br></h1>
</div>

You can retrain the instruction adapter of ChartVLM and reproduce our results via the following steps:

```
cd adapter/train
```
## Training data
We have provided our training data in `data.txt`  
You can also customize your own data with the following format:
```
[YOUR CUSTOMIZED SENTENCE 1] \t [LABEL 1]
[YOUR CUSTOMIZED SENTENCE 2] \t [LABEL 2]
......
```
Note that `[Label]` indicates the task type, which has been defined as:
```
0: Structural Extraction
1: Description/Summarization
2: Chart Type
3: Chart Title
4: Redrawing
5: Question Answering
```

## Train 
* You can train the instruction adapter using the following command:

```
python train_adapter.py --data_path ./path/to/data.txt --save_model_path ./path/to/model
```