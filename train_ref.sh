
#export GLOO_SOCKET_IFNAME='ens121f0'
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export DETECTRON2_DATASETS='/data/zqc/datasets/ref'
#SWIN_PATH=gres_swin_base.pth
#OUTPUT_PATH=./outputs/tem1


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net_v2.py \
#    --config-file configs/referring_swin_base.yaml \
#    --num-gpus 8 --dist-url auto \
#    OUTPUT_DIR ${OUTPUT_PATH}
export TORCH_DISTRIBUTED_DEBUG=DETAIL

DATASET=refcocog
DATA_PATH=/data/sunjiamu/home/data/images
REFER_PATH=/data/zqc/datasets/ref
MODEL=gres
SWIN_PATH=swin_base_patch4_window7_224_22k.pth
BERT_PATH=/data/zqc/huggingface-models/bert-base-uncased
RESUME=/data/zqc/code/ReLA_merge/outputs/msdatten_extract_supp_tokens_caris_ref/model_best_refcoco.pth
IMG_SIZE=480
now=$(date +"%Y%m%d_%H%M%S")
condition=mask
model_name='msdatten_extract_supp_tokens_ref_70'
OUTPUT_PATH=./outputs/msdatten_extract_supp_tokens_ref_70_v2
CONFIG=./configs/ref_swin_base2.yaml
EVAL=0
mkdir -p ${OUTPUT_PATH}
mkdir -p ${OUTPUT_PATH}/${DATASET}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 6224 train_net_ref.py --model ${MODEL} \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size  24 --pin_mem --print-freq 100 --workers 8 --config-file ${CONFIG}  \
        --lr 1e-4 --wd 1e-2 --swin_type base --condition ${condition} --model_name ${model_name}  \
        --warmup --warmup_ratio 1e-3 --warmup_iters 6620 --clip_grads --clip_value 0.01 \
        --pretrained_swin_weights ${SWIN_PATH} --epochs 75 --img_size ${IMG_SIZE} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} --output-dir ${OUTPUT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}'/'${DATASET}'/'train-${now}.txt