"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import torch as th
import torch.distributed as dist
import autorootcwd
from src.data.image_datasets import load_data
from src.training import dist_util
from src.loggings import logger
from src.scripts.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

NUM_CLASSES = 1


def main():
    args = create_argparser().parse_args()
    output_dir = f"{args.dataset}-{args.prior_model}"
    epoch = args.model_path.split('_')[-1].split('.')[0]
    result_dir = f"{output_dir}/results-{epoch}"
    os.makedirs(result_dir, exist_ok=True)
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    dataloader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        model=args.prior_model,
        mode='test',
    )

    logger.log("sampling...")
    
    loader = tqdm(dataloader, desc="Sampling batches", unit="batch")
    for i, (batch, cond, _) in enumerate(loader):
        loader.set_description(f"sampling batch {i+1}")
        model_kwargs = {"img": batch.to(dist_util.dev())}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        for j in range(batch.shape[0]):
            plt.imsave(f"{result_dir}/{i}_input_{j}.png", batch[j,0,:,:].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_output_{j}.png", sample[j,:,:,0].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_image_{j}.png", cond[j,0,:,:].cpu().numpy(), cmap='gray')


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        dataset="OCTA500_3M",
        data_dir="",
        clip_denoised=True,
        # num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="",
        gpu="0",
        prior_model='FRUnet',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
