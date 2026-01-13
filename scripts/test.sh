#!/bin/bash
set -e

export DATASET="OCTA500_6M"
export IMAGE_SIZE=256
export BS=16
export EPOCH=100000
export GPU=0
############################################################################
export DIFFUSION_TYPE="poisson"
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
    uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed on GPU 0."
) &
    
(
    export GPU=1
    export MODEL=CSNet
    eval "$METHOD_POISSON"

    export MODEL=SwinUNETR
    eval "$METHOD_POISSON"
    uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed on GPU 1."
) &



# uv run scripts/create_table.py --dataset "$DATASET" --epochs 100000
############################################################################
export DIFFUSION_TYPE="binomial"


METHOD_BERDIFF="uv run image_sample.py --dataset \$DATASET --data_dir data/\$DATASET \
    --model_path workdir/\$DIFFUSION_TYPE-mse-concat/OCTA500_6M-FRUnet/checkpoints/ema_0.9999_\$EPOCH.pt \
    --image_size \$IMAGE_SIZE --num_channels 128 --num_res_blocks 3 \
    --diffusion_steps 1000 --noise_schedule linear --batch_size \$BS --prior_model \$MODEL \
    --gpu \$GPU --diffusion_type \$DIFFUSION_TYPE \
     --ltype mse --image_encoder False --refine False"


(
    export GPU=2
    export MODEL=FRUnet
    eval "$METHOD_BERDIFF"

    export MODEL=Unet
    eval "$METHOD_BERDIFF"
    uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed on GPU 2."
) &
    
(
    export GPU=3
    export MODEL=CSNet
    eval "$METHOD_BERDIFF"

    export MODEL=SwinUNETR
    eval "$METHOD_BERDIFF"
    uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed on GPU 3."
) &

wait

# uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$EPOCH completed."

uv run scripts/create_table.py --dataset "$DATASET" --epochs 100000
