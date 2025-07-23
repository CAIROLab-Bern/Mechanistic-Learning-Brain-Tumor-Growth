"""
Train a diffusion model on images.
This file was taken from the original codebase for the paper "Denoising Diffusion Models for Medical Anomaly Detection" by Wolleb et al.
Minimal changes were done.
"""

import sys
import argparse
import torch as th
import torch.nn as nn
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import yaml

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.results_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if th.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False, validation_flag = args.validation_flag, single_contrast=args.single_contrast)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
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
        dataset=args.dataset
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        results_dir='./results',
        device = 'cluster',
        single_contrast = False,
        config = './scripts/config/config_diff_train.yaml'
    )
    defaults.update(model_and_diffusion_defaults())
    #import the config file and update with those
    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        defaults.update(config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
