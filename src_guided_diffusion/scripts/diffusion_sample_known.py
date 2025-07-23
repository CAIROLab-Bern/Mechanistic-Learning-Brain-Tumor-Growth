"""
Generate a 'diffused' version of an MRI scan, starting from a baseline image. 
Identical to guided diffusion, where the sample is known, but no additional guidance is enforced.  
"""
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import pickle
import yaml

#define custom iterator to be able to return current and enxt element for getting the label
class CustomIterator:
    def __init__(self, iterable):
        self.iterable = iter(iterable)
        self.current_element = None
        self.next_element = None
        self._advance()

    def _advance(self):
        self.current_element = self.next_element
        try:
            self.next_element = next(self.iterable)
        except StopIteration:
            self.next_element = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_element is None and self.next_element is None:
            raise StopIteration
        self._advance()
        return self.current_element

    def __current__(self):
        if self.current_element is None:
             self._advance()
        if self.current_element is None:
            raise StopIteration
        return self.current_element

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.results_dir)


    ds = BRATSDataset(args.data_dir, test_flag=args.test_flag, validation_flag=args.validation_flag, partition=args.partition, partition_number = args.partition_number, single_contrast = args.single_contrast, device=args.device, list_patients=args.list_patients)
    datal = th.utils.data.DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False)
    
    # Load the model's state_dict
    state_dict = th.load(args.model_path, map_location=dist_util.dev())

    # Rename the keys in the state_dict
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # remove 'module.' prefix from key
        new_state_dict[new_key] = value



    logger.log("loading model...")
    model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())


    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

        
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    print('pmodel ', p1)

    #still need this function, normally it serves for the addition of the regressor gradient, but here I am only looking to sample using DDIM without the regressor guidance
    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            a = x
            a=th.autograd.grad(x_in.sum(), x_in)[0]
            return  a, a * args.regressor_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    #define list to hold samples, anomaly maps and patient identifiers
    src_samples_list = []
    #samples array
    all_samples=[]
    #anomaly array 
    all_anomalies=[]
    #patient identifiers
    patient_ids=[]

    image_count = 0

    for img in datal:
        model_kwargs = {}
        if args.dataset=='brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number=img[4][0]

            #condition for taking only images from one of the classes
            if img[2]==args.sel_class_label:
                continue    
            image_count=image_count+1
            if image_count<=400:
                continue

            #add the patient number to patient ids
            patient_no = img[4][0].split("'")[0]
            patient_ids.append(patient_no)

        if args.class_cond:
            if img[2]==0:
                classes = th.randint(low=0, high=1, size=(args.batch_size,), device=dist_util.dev())
            else:
                classes = th.randint(low=1, high=2, size=(args.batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
            print('y', model_kwargs["y"]) 
            
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level
        )

        if args.dataset=='brats':
            if not args.single_contrast:
                all_samples.append(sample.cpu().numpy().reshape((4,256,256)))
            else:
                all_samples.append(sample.cpu().numpy().reshape((1,256,256)))
            difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)        
            all_anomalies.append(difftot.cpu().numpy())


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        
        if (image_count%100==0):
            src_samples_list=[all_samples, all_anomalies, patient_ids]
            with open(f'{args.results_dir}/diff_samples_list_{image_count}.pkl', 'wb') as f:
                pickle.dump(src_samples_list, f)
        
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    src_samples_list=[all_samples, all_anomalies, patient_ids]

    with open(f'{args.results_dir}/diff_samples_list_final.pkl', 'wb') as f:
        pickle.dump(src_samples_list, f)

    dist.barrier()
    logger.log("Sampling Complete")

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        model_path="",
        noise_level=500,
        dataset='brats', 
        results_dir = './results',
        partition = False,
        partition_number = 0.1, 
        single_contrast = False,
        device='cluster',
        regressor_scale =0,
        use_ddim=True,
        sel_class_label=1,
        list_patients="",
        test_flag = False,
        validation_flag = True,
        config='./scripts/config/config_diff_sample.yaml'
    )
    defaults.update(model_and_diffusion_defaults())
    #import the config file and update with those
    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        print(config)
        defaults.update(config)
    print(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

