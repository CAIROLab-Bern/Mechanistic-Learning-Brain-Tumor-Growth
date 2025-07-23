"""
Train a regressor (including time embeddings) for predicting relative tumor size.
"""

import argparse
import os
import sys
from torch.autograd import Variable
import torch.nn as nn
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import blobfile as bf
import torch as th
os.environ['OMP_NUM_THREADS'] = '8'
from sklearn.metrics import mean_squared_error
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from visdom import Visdom
import numpy as np
import pandas as pd

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize, parse_resume_step_from_filename, log_loss_dict
from guided_diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
)
import yaml


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(args.results_dir)
    logger.log("creating model and diffusion...")
    model, diffusion = create_regressor_and_diffusion(**args_to_dict(args, regressor_and_diffusion_defaults().keys()),)
    model.to(dist_util.dev())

    #create scheduler for time embeddings
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())
    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.regressor_use_fp16, initial_lg_loss_scale=16.0
    )


    logger.log("creating data loader...")
    #enable different data loader depending on the dataset
    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False, single_contrast=args.single_contrast, device=args.device,  regression=args.regression, type_slices = args.type_slices)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=False)
        data = iter(datal)



    logger.log(f"creating optimizer...")
    opt = AdamW(filter(lambda p: p.requires_grad, mp_trainer.master_params), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt")
        print(bf.dirname(args.resume_checkpoint))
        print(f"opt{resume_step:06}.pt")
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev()))

    ##### FUNCTIONS FOR TRAINING AND VALIDATION #####
    def forward_backward_log(data_loader, step, prefix="train"):
        if args.dataset=='brats':
            batch, extra, labels,_ , _ = next(data_loader)
        batch = batch.to(dist_util.dev())
        labels= labels.to(dist_util.dev())

        #if to include time embeddings
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)):
          
            sub_batch = Variable(sub_batch, requires_grad=True)
            preds = model(sub_batch, timesteps=sub_t)
            #reshape the preds tensor 
            preds = preds.view(args.batch_size,)
            preds = preds.to(dist_util.dev())
            loss = F.mse_loss(preds, sub_labels.float(), reduction="none")
            
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            log_loss_dict(diffusion, sub_t, losses)

            loss = loss.mean()
            if prefix!="train":
                output_idx = preds[0].argmax()
                print('outputidx', output_idx)
                output_max = preds[0, output_idx]
                print('outmax', output_max, output_max.shape)
                output_max.backward()
                saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
                print('saliency', saliency.shape)
                th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
        return losses

    def run_validation(data_dir_vals):
        """Helper function to run validation while training the regressor.

        Arguments:
            data_dir_vals {str} -- path/(s) to the validation directory/(ies)
        """
        logger.log("Running validation on model...")

        data_dir_vals = data_dir_vals.split(",")

        #if more than one dataset path is provided for validation, perform separate validation for all of them (e.g. adult and pediatric)
        if len(data_dir_vals)>1:
            for data_dir_val in data_dir_vals:
                #skeleton based on the script in regression test
                if args.dataset == 'brats':
                    ds_val = BRATSDataset(data_dir_val, test_flag=True, single_contrast=args.single_contrast, device=args.device,  regression=args.regression)
                    datal_val = th.utils.data.DataLoader(
                        ds_val,
                        batch_size=1,
                        shuffle=True)
                    data_val = iter(datal_val)

                total = 0
                labels_all=[]
                preds_all=[]
                number_all=[]
                count=0
                with th.no_grad():
                    for batch, extras, labels, _,number in data_val:
                        if labels.cpu().detach().numpy().ravel()==0:
                            print("We only validate on diseased slices.") #depending on application
                            continue
                        batch = batch.to(dist_util.dev())
                        labels = labels.to(dist_util.dev())
                        labels_all.append(np.array(labels.cpu()))
                        number_all.append(np.array(number))
                        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
                        sub_batch = Variable(batch, requires_grad=False)
                        sub_batch = sub_batch.to(dist_util.dev())
                        preds = model(sub_batch, timesteps=t)
                        preds_all.append(np.concatenate(np.array(preds.cpu())))
                        total += len(batch)
                        count=count+1

                preds_array= np.concatenate(preds_all)
                labels_array=np.concatenate(labels_all)

                mse = mean_squared_error(labels_array, preds_array)
                print("The overall mean squared error (MSE) is: {:.3}".format(mse))
                dataset = data_dir_val.split("/")[-2]
                if not os.path.exists(f"{args.results_dir}/validation.csv"):
                    validation_csv = pd.DataFrame({'iterations': step+resume_step,'set':dataset, 'mse': mse}, index=[0])
                else:
                    validation_csv = pd.read_csv(f"{args.results_dir}/validation.csv")
                    validation_csv = validation_csv.append({'iterations':step+resume_step,'set':dataset,'mse': mse}, ignore_index=True)
                validation_csv.to_csv(f"{args.results_dir}/validation.csv", index=False)
                logger.logkv(f"val_step_{dataset}", step + resume_step)
                logger.logkv(f"val_mse_{dataset}", mse)
                logger.dumpkvs()
        logger.log("Ran validation on model!")

    
    ##### MAIN ########
    logger.log("training regressor model...")
    correct=0; total=0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        print('step', step + resume_step)
        
        try:
            losses = forward_backward_log(data, step + resume_step)
        except:
            data = iter(datal)
            losses = forward_backward_log(data, step + resume_step)

        total+=args.batch_size
        mp_trainer.optimize(opt)
          
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)
            run_validation(args.data_dir_val)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()




def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

def compute_top_k(preds, labels, k, reduction="mean"):
    _, top_ks = th.topk(preds, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        data_dir_val="./data/validation",
        dataset='brats',
        noised=True,
        lr=1e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=3,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=1,
        iterations=40000,
        eval_interval=1000,
        save_interval=5000,
        results_dir='./results',
        config='./scripts/config/config_reg_train.yaml'
        )
    defaults.update(regressor_and_diffusion_defaults())
    #import the config file and update with those
    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        defaults.update(config)
    #override if command line arguments are passed
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    return parser


if __name__ == "__main__":
    main()
