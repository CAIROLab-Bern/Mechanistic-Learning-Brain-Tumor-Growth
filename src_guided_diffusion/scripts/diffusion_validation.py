"""
Validation for a trained diffusion model. 
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
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8097)
import pandas as pd
import fcntl
import os
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

    
    avg_lossmse_train=0
    avg_lossmse_val=0   

    logger.log("validation...")
    logger.log("On the validation dataset")

    logger.log("creating data loader...")

    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir_val, test_flag=False, validation_flag = True, single_contrast=args.single_contrast)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)

    avg_lossmse_val = TrainLoop(
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
    ).run_validation()

    print("I finished validation on the validaiton data.")
    print("This is the final loss ", avg_lossmse_val)

    logger.log("On the training dataset")
    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir_train, test_flag=False, validation_flag = False, single_contrast=args.single_contrast, partition=args.partition, partition_number=args.partition_number)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)

    avg_lossmse_train = TrainLoop(
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
    ).run_validation()

    print("I finished validation on the training data.")
    print("This is the final loss ", avg_lossmse_train)

    #saving results 
    model_name = args.resume_checkpoint.split("/")[-2]
    log_results_path = args.log_results+f"/{model_name}.csv"

    if not os.path.exists(log_results_path):
        #create data frame 
        #get all of the saved checkpoints 
        models=[]
        for root, dirs, files in os.walk(os.path.dirname(args.resume_checkpoint)):
            for f in files:
                if "model" in f:
                    model_checkpoint = f.split(".pt")[0].split("model")[1]
                    models.append(model_checkpoint)
        
        index_values = ["BraTS23_train","BraTS23_val","BraTS23_PED_train","BraTS23_PED_val"]
        results_df=pd.DataFrame(columns=models, index=index_values)
    else:
        results_df = pd.read_csv(log_results_path, index_col=0)

    current_model = args.resume_checkpoint.split("/")[-1].split(".pt")[0].split("model")[1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    dataset = args.data_dir_val.split("/")[-2]
    results_df.loc[str(f"{dataset}_val")][str(current_model)] = avg_lossmse_val.cpu().detach().numpy()
    results_df.loc[str(f"{dataset}_train")][str(current_model)] = avg_lossmse_train.cpu().detach().numpy()

    results_df.to_csv(log_results_path)
    print("Saved file.")
        

def create_argparser():
    defaults = dict(
        data_dir_val="./data/validation",
        data_dir_train="./data/train",
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
        validation_flag = False, 
        device = 'cluster',
        remove_small = False,
        partition = False,
        partition_number = 0.2,
        log_results = "./results",
        config = './scripts/config/config_diff_val.yaml'
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
