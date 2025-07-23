import matplotlib.pyplot as plt
import torch as th
import argparse
import os
import pickle 
import numpy as np
import pandas as pd 
import nibabel
from sklearn.metrics import roc_curve, auc
from scipy.special import expit
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append("..")
sys.path.append(".")
from torch.autograd import Variable
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util
from utils.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    NUM_CLASSES
)
#import mean squared error
from sklearn.metrics import mean_squared_error
import yaml

def main():
    args = create_argparser().parse_args()

    print("Creating model ...")

    model, diffusion = create_regressor_and_diffusion(
        **args_to_dict(args, regressor_and_diffusion_defaults().keys()),)
    model.to(dist_util.dev())

    print("Loading model...")
    model.load_state_dict(dist_util.load_state_dict( args.regressor_path, map_location=dist_util.dev()))

    if args.dataset=='brats':
        ds = BRATSDataset(args.data_dir, test_flag=args.test_flag, single_contrast = args.single_contrast, device=args.device, regression = True)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False)

    correct = 0
    total = 0
    labels_all=[]
    preds_all=[]
    number_all=[]
    count=0

    with th.no_grad():
        for batch,extras, labels, _,number in datal:
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
    

    number_array = np.concatenate(number_all)
    preds_array= np.concatenate(preds_all)
    labels_array=np.concatenate(labels_all)

    mse = mean_squared_error(labels_array, preds_array)
    print("The overall mean squared error is: {:.3}".format(mse))
    save_csv(number_array, labels_array, preds_array, mse, args.saving_name)

       
                
def save_csv(number, labels, preds, mse, saving_name):

    args = create_argparser().parse_args()

    if args.save_details:
        data = {
        'patient_slice': number,
        'label': labels,
        'predicted_label': preds
        }
        df = pd.DataFrame(data)
        # Save DataFrame as a CSV file
        df.to_csv(args.regressor_path.split("/")[-2]+"_"+args.regressor_path.split("/")[-1].split(".")[0]+f"_mse_{mse}.csv", index=False)

    iterations = args.regressor_path.split("/")[-1].split(".")[0].split("model")[1]
    dataset = args.data_dir.split("/")[-1]
    print(iterations)
    #save to progress folder
    if not os.path.isdir(args.regressor_path.split("/")[-2]+"_progress"):
        os.mkdir(args.regressor_path.split("/")[-2]+"_progress")

    progress = pd.DataFrame({'iterations':iterations,'set':dataset, 'mse': mse}, index=[0])
    progress.to_csv(saving_name, index=False)


    
def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        log_interval=1,
        dataset='brats',
        results_dir='./results',
        regressor_path = '',
        device = 'cluster',
        partition=False,
        partition_size=0.1,
        single_contrast =False,
        regression = True,
        save_details = True,
        saving_name = '',
        config='./scripts/config/config_reg_test.yaml',
        test_flag = True
    )
    defaults.update(regressor_and_diffusion_defaults())
    #import the config file and update with those
    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        defaults.update(config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()