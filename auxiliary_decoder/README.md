<div align="center">
<h1>Auxiliary decoder for ChartVLM<br></h1>
</div>

You can fine-tune the auxiliary decoder of ChartVLM from pre-trained Vicuna via the following steps:

```
cd auxiliary_decoder/train
```
## Training data
We have provided a template of training data in `./templates/train_data.json`  
You can also customize your own data with the following format:
```
[
    { 
      "instruction":  xxxxxxxxx
      "input": xxxxxxxxx
      "output": xxxxxxxxx
    },
    ......
]
```
Note that all the data we used for fine-tunning in ChartVLM are from ChartQA, PlotQA, Chart2Text and SimChart9K.


## Train 
* You can fine-tune the auxiliary decoder using one single GPU:

```
python finetune.py \
--base_model ${PATH_TO_PRETRAINED_VICUNA} \
--output_dir ${PATH_TO_OUTPUT_DIR} \
--num_epochs 5 \
--lora_r 16 \
--lora_alpha 32 \
--data_path ${PATH_TO_TRAIN_DATA}$
```

* or using multiple GPUs:
```
bash scripts/dist_train.sh ${NUM_GPUS} \
--base_model ${PATH_TO_PRETRAINED_VICUNA} \
--output_dir ${PATH_TO_OUTPUT_DIR} \
--num_epochs 5 \
--lora_r 16 \
--lora_alpha 32 \
--data_path ${PATH_TO_TRAIN_DATA}$
```
