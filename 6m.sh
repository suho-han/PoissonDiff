uv run image_train.py --dataset OCTA500_6M --data_dir data/OCTA500_6M \
--image_size 256 --num_channels 128 --num_res_blocks 3 \
--diffusion_steps 1000 --noise_schedule linear \
--lr 1e-4 --batch_size 8 --prior_model FRUnet --gpu "1"

python scripts/alert.py --repo PoissonDiff --message "Training on OCTA500_6M with FRUnet completed."

uv run image_sample.py --dataset OCTA500_6M --data_dir data/OCTA500_6M --model_path OCTA500_6M-FRUnet/ema_0.9999_100000.pt \
--image_size 256 --num_channels 128 --num_res_blocks 3 \
--diffusion_steps 1000 --noise_schedule linear \
--batch_size 16  --prior_model FRUnet --gpu "1"

python scripts/alert.py --repo PoissonDiff --message "Sampling on OCTA500_6M with FRUnet completed."