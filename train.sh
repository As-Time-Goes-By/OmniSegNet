export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET='OmniRef'
DATA_PATH=./data/images
REFER_PATH=./datasets/OmniRef
MODEL=omnisegnet
SWIN_PATH=swin_base_patch4_window12_384_22k.pth
BERT_PATH=bert-base-uncased
IMG_SIZE=480
now=$(date +"%Y%m%d_%H%M%S")
condition=mask
model_name='OmniSegNet'
OUTPUT_PATH=./outputs/${model_name}
CONFIG=./configs/referring_swin_base.yaml
mkdir -p ${OUTPUT_PATH}
mkdir -p ${OUTPUT_PATH}/${DATASET}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 6224 train_net.py --model ${MODEL} \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size  8 --pin_mem --print-freq 100 --workers 8 \
        --lr 1e-4 --wd 1e-2 --swin_type base --condition ${condition} --model_name ${model_name} --bert_tokenizer ${BERT_PATH} \
        --warmup --warmup_ratio 1e-3 --warmup_iters 7500 --clip_grads --clip_value 0.01 --output-dir ${OUTPUT_PATH} \
        --config-file ${CONFIG} --epochs 50 --img_size ${IMG_SIZE} --pretrained_swin_weights ${SWIN_PATH} \
        --image_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}'/'${DATASET}'/'train-${now}.txt
