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

from src.data.image_datasets import load_data
from src.loggings import logger
from src.loggings.run_history import append_run_history
from src.training import dist_util
from src.utils.patch_sampling import patch_sample
from src.utils.script_util import add_dict_to_argparser, args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults
from src.utils.tensor_io import save_tensor_as_npy

NUM_CLASSES = 1


def main():
    args = create_argparser().parse_args()
    base_output = "test-run" if args.test_run else "workdir"
    output_dir = f"{base_output}/{args.diffusion_type}/{args.dataset}-{args.prior_model}"
    epoch = args.model_path.split('_')[-1].split('.')[0]
    result_dir = f"{output_dir}/results-{epoch}"
    os.makedirs(result_dir, exist_ok=True)
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=output_dir)

    append_run_history(
        args,
        mode="sample",
        command="image_sample",
        epoch=epoch,
        output_dir=output_dir,
        result_dir=result_dir,
    )

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
        batch_size = image.shape[0]
        model_kwargs = {
            "img": image.to(dist_util.dev()),
            "prior": input.to(dist_util.dev()),
        }
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim
            else diffusion.sample if args.diffusion_type == "poisson_flow"
            else diffusion.ddim_sample_loop
        )

        with torch.inference_mode():
            if image.shape[2] > args.image_size or image.shape[3] > args.image_size:
                sample, intermediates = patch_sample(
                    sample_fn=sample_fn,
                    model=model,
                    image=image.to(dist_util.dev()),
                    prior=input.to(dist_util.dev()),
                    input_size=args.image_size,
                    patches_per_dim=args.patches_per_dim,
                    model_kwargs=model_kwargs,
                )
            else:
                sample, intermediates = sample_fn(
                    model,
                    (batch_size, 1, args.image_size, args.image_size),
                    # clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    return_intermediates=True,
                )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        final_output = (sample > 0.5).float()
        final_output = final_output.to(torch.uint8)
        os.makedirs(f"{result_dir}/intermediates", exist_ok=True)
        for j in range(input.shape[0]):
            save_tensor_as_npy(f"{result_dir}/{i}_image_{j}.npy", image[j])
            save_tensor_as_npy(f"{result_dir}/{i}_target_{j}.npy", target[j])
            save_tensor_as_npy(f"{result_dir}/{i}_output_{j}.npy", sample[j])
            save_tensor_as_npy(f"{result_dir}/{i}_final_output_{j}.npy", final_output[j])
            save_tensor_as_npy(f"{result_dir}/{i}_input_{j}.npy", input[j])

            for step, out in enumerate(intermediates):
                save_tensor_as_npy(
                    f"{result_dir}/intermediates/{i}_image_{j}_step_{step}",
                    out[j],
                )
        if args.test_run:
            break
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
        patches_per_dim=2,  # Number of patches per dimension for large images
        test_run=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
