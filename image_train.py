"""
Train a diffusion model on images.
"""

import argparse
import autorootcwd
from src.training import dist_util
from src.loggings import logger
from src.data.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from src.scripts.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.training.train_utils import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.gpu)
    output_dir = f"{args.dataset}-{args.prior_model}"
    logger.configure(dir=output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        model=args.prior_model,
        mode='train',
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        dataloader=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        output_dir=output_dir,
    ).run_loop(step_limit=100000)


def create_argparser():
    defaults = dict(
        dataset="OCTA500_3M",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu="0",
        prior_model='FRUnet',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
