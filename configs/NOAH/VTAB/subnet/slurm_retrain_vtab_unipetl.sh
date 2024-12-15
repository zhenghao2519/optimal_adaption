#!/bin/sh

#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH -J VTAB-RT
#SBATCH --time=01:00:00

#SBATCH -o file_to_redirect_the_std_output.out
#SBATCH -e file_to_redirect_the_std_err.err

# set -x 

currenttime=$(date "+%Y%m%d_%H%M%S")

PARTITION='lrz-hgx-h100-92x4'
# NODE='DESKTOP-FV3EM2A'
JOB_NAME=VTAB-RT
GPUS=1
CKPT='../pretrained/ViT-B_16.npz' #$1
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

RANDOM=42

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


# METHOD=ROUTER
METHOD=GATE # the first 2 gate are router with 0.001 and 0.0005

for LR in 0.0005; do
# for LR in 0.002 0.003; do
    for DATASET in cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele; do #
    # for DATASET in dtd oxf  ord_flowers102 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_ori smallnorb_azi smallnorb_ele; do #
    # for DATASET in oxford_flowers102 clevr_count; do
    # for DATASET in cifar100; do
        export MASTER_PORT=$((12000 + $RANDOM % 20000))
        srun -p ${PARTITION} \
            --exclude=lrz-hgx-h100-019 \
            --job-name=${JOB_NAME}-${DATASET} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=${GPUS} \
            --ntasks-per-node=${GPUS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --kill-on-bad-exit=1 \
            ${SRUN_ARGS} \
            python supernet_train_prompt.py --data-path=/dss/dsshome1/0C/ge74mip2/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=experiments/NOAH/subnet/VTAB_unipetl/ViT-B_prompt_unipetl.yaml --resume=${CKPT} --output_dir=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/retrain_${LR}_wd-${WEIGHT_DECAY} --batch-size=64 --mode=retrain --epochs=200 --lr=${LR} --weight-decay=${WEIGHT_DECAY} --no_aug --direct_resize --mixup=0 --cutmix=0 --smoothing=0 --launcher="none" --router --add_vpt_gate 2>&1 | tee -a logs/unipetl-${METHOD}-${currenttime}-${DATASET}-${LR}-vtab-rt.log >/dev/null &
        echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${LR}-vtab-rt.log\" for details. ]\033[0m"
    done
done
