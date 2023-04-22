# DTCA

## 1. dataset and pretrained models
Download twitter-2015 dataset and twitter-2017 dataset and put them into datasets/ .

Download pretrained models and put them into models/ .

## 2. environments
    pip install -r requirements.txt

If raise no module error, install module by yourself
## 3. generate inputs
    python utils/TrainInputProcess.py \
        --dataset_type '2015' \
        --text_model_type 'roberta' \
        --image_model_type 'vit'
Other parameters introduced in code. There are other preprocess for other use, no need to pay attention.


## 4. run model
    python main.py \
        --dataset_type '2015' \
        --text_model_type 'roberta' \
        --image_model_type 'vit'
    or
    sh run.sh
other training parameters introduced in code

## 5. model DTCA
Implementation in code model/modeling_dtca.py

## 6. ablation study
Set alpha = 0 or beta = 0

## Welcome any questions!


