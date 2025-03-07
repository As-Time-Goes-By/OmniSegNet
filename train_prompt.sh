
#export GLOO_SOCKET_IFNAME='ens121f0'
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export DETECTRON2_DATASETS='/data/zqc/datasets/ref'
#SWIN_PATH=gres_swin_base.pth
#OUTPUT_PATH=./outputs/tem1


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net_v2.py \
#    --config-file configs/referring_swin_base.yaml \
#    --num-gpus 8 --dist-url auto \
#    OUTPUT_DIR ${OUTPUT_PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET=gvpcoco
DATA_PATH=/data/sunjiamu/home/data/images
REFER_PATH=/data/zqc/datasets/ref
MODEL=gres
SWIN_PATH=swin_base_patch4_window7_224_22k.pth
BERT_PATH=/data/zqc/huggingface-models/bert-base-uncased
#OUTPUT_PATH=./outputs/merge3
IMG_SIZE=480
now=$(date +"%Y%m%d_%H%M%S")
condition=mask
model_name='msdatten_extract_supp_tokens_prompt_v3'
#OUTPUT_PATH=./outputs/${model_name}
RESUME='/data/zqc/code/ReLA_merge/outputs/msdatten_extract_supp_tokens_prompt_box/model_best_gvpcoco_box.pth'
OUTPUT_PATH=./outputs/msdatten_extract_supp_tokens_prompt_v3_no_src+mask
mkdir -p ${OUTPUT_PATH}
mkdir -p ${OUTPUT_PATH}/${DATASET}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 6284 train_net_v3.py --model ${MODEL} \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size  18 --pin_mem --print-freq 100 --workers 8 \
        --lr 1e-4 --wd 1e-2 --swin_type base --condition ${condition} --model_name ${model_name}  \
        --warmup --warmup_ratio 1e-3 --warmup_iters 7500 --clip_grads --clip_value 0.01 \
        --pretrained_swin_weights ${SWIN_PATH} --epochs 50 --img_size ${IMG_SIZE} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} --output-dir ${OUTPUT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}'/'${DATASET}'/'train-${now}.txt