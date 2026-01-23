#!/bin/bash
set -e

DATASET=OCTA500_6M
MODEL=FRUnet
EPOCH=1000000

DIFFUSION_TYPE="binomial"
GPU=1

uv run image_train.py --dataset "$DATASET" --data_dir data/"$DATASET" \
    --prior_model "$MODEL"  --image_size 256 --num_channels 128 --num_res_blocks 3 \
    --diffusion_steps 1000 --noise_schedule linear \
    --gpu "$GPU" --lr 1e-4 --batch_size 8  \
    --diffusion_type "$DIFFUSION_TYPE" --total_steps "$EPOCH" \
    --refine False --image_encoder False --ltype mse

uv run scripts/alert.py --repo PoissonDiff --message "Training on $DIFFUSION_TYPE/$DATASET/$MODEL completed."

uv run image_sample.py --dataset "$DATASET" --data_dir data/"$DATASET" \
    --prior_model "$MODEL"  --image_size 256 --num_channels 128 --num_res_blocks 3 \
    --diffusion_steps 1000 --noise_schedule linear \
    --gpu "$GPU" --batch_size 4  \
    --diffusion_type "$DIFFUSION_TYPE" \
    --model_path workdir/"$DIFFUSION_TYPE"/"$DATASET"-"$MODEL"/checkpoints/ema_0.9999_"$EPOCH".pt \
    --refine False --image_encoder False --ltype mse
    
uv run scripts/alert.py --repo PoissonDiff --message "Sampling on $DIFFUSION_TYPE/$DATASET/$MODEL completed."

uv run scripts/create_table.py --dataset "$DATASET" --model "$MODEL" --epoch "$EPOCH" --diffusion-type "$DIFFUSION_TYPE"

wait
