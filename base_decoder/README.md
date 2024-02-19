<div align="center">
<h1>Chart image encoder and base decoder for ChartVLM<br></h1>
</div>

You can retrain the base decoder of ChartVLM and reproduce our results via the following steps:

```
cd base_decoder/train
```
## Download dataset, we merge four datasets as follows
The following datasets are used in our paper: 
- ChartQA \[[Dataset Page](https://github.com/vis-nlp/ChartQA)\]
- PlotQA \[[Dataset Page](https://github.com/NiteshMethani/PlotQA)\]
- Chart2Text \[[Dataset Page](https://github.com/JasonObeid/Chart2Text)\]
- SimChart9K \[[Download](https://github.com/UniModal4Reasoning/SimChart9K)\]

## Preprocessing the downloaded chart data 
* In order to speed up the data i/o during the training process for base_decoder, we choose to preprocess the downloaded chart data, saving as the .npy format.

* You have to first preprocess the data before starting the training process (This should be the absolute path of the downloaded datasets)
```
cd tools/data_preprocess/
# Change the root path for the downloaded ChartQA or Chart2Text dataset
python data_preprocess_ChartQA_Chart2Text.py
```

```
cd tools/data_preprocess/
# Change the root path for the downloaded PlotQA dataset
python data_preprocess_PlotQA.py
```

```
cd tools/data_preprocess/
# Change the root path for the downloaded SimChart9K dataset
python data_preprocess_SimChart9K.py
```

```
# return to the 'tools' directory
cd ..
```

## Training and tesing the base decoder for ChartVLM
* Train the Base Model using multi-GPU
```shell script
sh scripts/dist_train.sh 8 \
--config ./cfgs/image_to_csv_base_merge_all_trained.yaml \
--VAL_PER_EPOCH 0
```

* Train the Base Model using multi-machines
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/image_to_csv_base_merge_all_trained.yaml \
--VAL_PER_EPOCH 0
```

* Train the Large Model using multi-GPU
```shell script
sh scripts/dist_train.sh 8 \
--config ./cfgs/image_to_csv_large_merge_all_trained.yaml \
--VAL_PER_EPOCH 0
```

* Train the Large Model using multi-machines
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/image_to_csv_large_merge_all_trained.yaml \
--VAL_PER_EPOCH 0
```

* Evaluate the Model using multi-GPU, SCRM metric, and 1280 output tokens
```shell script
sh scripts/dist_test.sh 4 \
--config ./cfgs/image_to_csv_large_merge_all_trained.yaml \
--criterion csv_metric \
--num_token 1280
```