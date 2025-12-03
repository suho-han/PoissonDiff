"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import math
import os

import autorootcwd
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from tqdm import tqdm

from scripts.create_table import evaluate_results
from src.data.image_datasets import load_data
from src.loggings import logger
from src.scripts.patch_sampling import patch_sample
from src.scripts.script_util import add_dict_to_argparser, args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults
from src.training import dist_util

NUM_CLASSES = 1


def main():
    args = create_argparser().parse_args()
    output_dir = f"workdir/{args.diffusion_type}-{args.dataset}-{args.prior_model}"
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
        deterministic=True,
    )

    logger.log(f"sampling {len(dataloader.dataset)} images...")
    num_batches = math.ceil(len(dataloader.dataset) / args.batch_size)
    loader = tqdm(dataloader, desc="Sampling batches", unit="batch")
    for i, (input, target, image) in enumerate(loader):
        loader.set_description(f"sampling batch {i+1}/{num_batches}")
        model_kwargs = {"img": image.to(dist_util.dev()),
                        "prior": input.to(dist_util.dev())}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Use patch sampling if image is larger than model input size
        if image.shape[2] > args.image_size or image.shape[3] > args.image_size:
            sample, intermediates = patch_sample(
                sample_fn=sample_fn,
                model=model,
                image=image.to(dist_util.dev()),
                prior=input.to(dist_util.dev()),
                input_size=args.image_size,
                target_size=image.shape[2],
                overlap=args.image_size//2,
                model_kwargs=model_kwargs,
                batch_size=args.batch_size,
            )
        else:
            sample, intermediates = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                # clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                return_intermediates=True,
            )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        final_output = (sample > 0.5).float()
        final_output = final_output.to(torch.uint8)*255
        os.makedirs(f"{result_dir}/intermediates", exist_ok=True)
        for j in range(input.shape[0]):
            plt.imsave(f"{result_dir}/{i}_image_{j}.png", image[j, 0, :, :].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_target_{j}.png", target[j, 0, :, :].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_output_{j}.png", sample[j, :, :, 0].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_final_output_{j}.png", final_output[j, :, :, 0].cpu().numpy(), cmap='gray')
            plt.imsave(f"{result_dir}/{i}_input_{j}.png", input[j, 0, :, :].cpu().numpy(), cmap='gray')

            for step, out in enumerate(intermediates):
                interm_sample = out[j].permute(1, 2, 0).contiguous()
                plt.imsave(f"{result_dir}/intermediates/{i}_image_{j}_step_{step}.png", interm_sample[:, :, 0].cpu().numpy(), cmap='gray')
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
        gpu="2",
        prior_model='FRUnet',
        stride=64,  # Stride for patch-based sampling of large images
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
