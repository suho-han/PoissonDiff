#!/bin/bash
set -e

export DATASET="OCTA500_6M"
export IMAGE_SIZE=256
export BS=64
export EPOCH=100000
export GPU=0
############################################################################
export DIFFUSION_TYPE="gaussian"
METHOD_POISSON="uv run image_sample.py --dataset \$DATASET --data_dir data/\$DATASET \
    --model_path workdir/\$DIFFUSION_TYPE-mse-concat/OCTA500_6M-FRUnet/checkpoints/ema_0.9999_\$EPOCH.pt \
    --image_size \$IMAGE_SIZE --num_channels 128 --num_res_blocks 3 \
    --diffusion_steps 1000 --noise_schedule linear --batch_size \$BS --prior_model \$MODEL \
    --gpu \$GPU --diffusion_type \$DIFFUSION_TYPE \
    --ltype mse --image_encoder False --refine False"

(
    export MODEL=FRUnet
    eval "$METHOD_POISSON"

    export MODEL=Unet
    eval "$METHOD_POISSON"

    export MODEL=CSNet
    eval "$METHOD_POISSON"

    export MODEL=SwinUNETR
    eval "$METHOD_POISSON"
    uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed on GPU 0."
) &

wait

# uv run scripts/create_table.py --dataset "$DATASET" --epochs 500000
############################################################################

uv run scripts/create_table.py --dataset "$DATASET" --epochs 100000
