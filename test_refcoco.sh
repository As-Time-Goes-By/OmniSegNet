#!/bin/bash
uname -a
#date
#env
date

DATASET=refcocog
DATA_PATH=/data/huangxiaorui/images
REFER_PATH=/data/zqc/datasets/ref
BERT_PATH=/data/zqc/huggingface-models/bert-base-uncased
MODEL=gres
SWIN_TYPE=base
IMG_SIZE=480
condition=mask
model_name='msdatten_extract_supp_tokens_ref_70'
ROOT_PATH=./outputs/${model_name}
RESUME_PATH=/data/zqc/code/ReLA_merge/outputs/msdatten_extract_supp_tokens_ref_70_v2/model_best_refcocog.pth
#OUTPUT_PATH=${ROOT_PATH}/${DATASET}
OUTPUT_PATH=./outputs/msdatten_extract_supp_tokens_ref_70_v2/${DATASET}
SPLIT=val
CONFIG=./configs/ref_swin_base2.yaml
#cd YOUR_CODE_PATH
python eval_v1.py --model ${MODEL} --swin_type ${SWIN_TYPE} \
        --dataset ${DATASET} --split ${SPLIT} --batch-size 32 --condition ${condition} --model_name ${model_name}\
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} --config-file ${CONFIG} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}/eval-${SPLIT}.txt
