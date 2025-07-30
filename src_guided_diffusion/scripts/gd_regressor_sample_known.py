"""
Inference script - denoising diffusion with regressor guidance. 
Partially noising the starting image and denoising with guidance towards a different tumor size. 
The target relative size change can be defined as a fixed value (e.g., 25% = 0.25) or by providing a dictionary with the sample names and respective relative changes.
Results are saved as an array including a list of generated images, anomalies (differences between orginal and generated) and the sample ID (or patient number).  
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.dipgloader import DIPGDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    regressor_defaults,
    classifier_defaults,
    create_regressor,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pathlib import Path
import nibabel
import pickle
import time
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

    #create folder to save inference results within the data directory
    saving_dir = f"{Path(args.data_dir).parent}/{args.run_id}"
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    print(f"Saving samples to {saving_dir}.")


    #if we don't have constant change we have a dictionary specifying the change
    if not args.constant_relative_change:
        print("Tumor change is defined within a dictionary")
        assert args.relative_tumor_change_dict!='', "Constant relative change is not enabled. Enable constant relative tumor change or provide a dictionary specifying change for every sample!"
        relative_tumor_change_dict = pickle.load(open(args.relative_tumor_change_dict, "rb"))
        #set the list of patients to the patients in that dictionary
        if args.list_patients=="":
        #if the list of patients is not specified, we take the keys from the dictionary
            print("The list of patients is not specified, taking the keys from the dictionary")
            args.list_patients = list(relative_tumor_change_dict.keys())
            print("The list of patients is: ", args.list_patients)


    if args.dataset=='brats':
      ds = BRATSDataset(args.data_dir, test_flag=args.test_flag, partition=args.partition, partition_number = args.partition_number, single_contrast = args.single_contrast, remove_small=args.remove_small, device=args.device, regression = True, list_patients=args.list_patients)
      datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    # Load the model's state_dict
    state_dict = th.load(args.model_path, map_location=th.device('cpu'))
    # Rename the keys in the state_dict
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # remove 'module.' prefix from key
        new_state_dict[new_key] = value

    #if generating probability maps, 100 generations with different random seeds are run
    if args.probability_map:
        generation_number = args.generation_number
        prob_maps_img_list = []
        prob_maps_anomaly_list = []
        prob_maps_patients_list = []
    else:
        generation_number = 1

    logger.log("loading regressor...")
    regressor = create_regressor(**args_to_dict(args, regressor_defaults().keys()))
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path , map_location="cpu"), strict=False
    )
    print('loaded regressor')
    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()

    all_samples_generations = []
    #running inference multiple times with different random seeds, if we want to generate probability maps, otherwise the one inf round will be ran with random seed = 0
    for random_seed in range(0,generation_number):
        
        start_time_generation = time.time()
        args.random_seed = random_seed
        print("The rendom seed for this generation is ",random_seed)
        logger.log("creating model and diffusion...")
        start_time = time.time()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        logger.log("loading model...")
        model.load_state_dict(new_state_dict)
        model.to(dist_util.dev())

        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print("Elapsed Time (GPU):", elapsed_time)

        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        
        p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
        p2 = np.array([np.array(p.shape).prod() for p in regressor.parameters()]).sum()
        print('pmodel', p1, 'pclass', p2)

        #if we are adding an additional gradient 
        if args.classifier_path:
            logger.log("loading classifier...")
            classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
            classifier.load_state_dict(
                dist_util.load_state_dict(args.classifier_path , map_location="cpu")
            )
            print('loaded classifier')
            p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
            p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
            print('pmodel', p1, 'pclass', p2)
            classifier.to(dist_util.dev())
            if args.classifier_use_fp16:
                classifier.convert_to_fp16()
            classifier.eval()

        

        #extracting the labels, we will use this to guide the tumors to a size relative to the starting one
        labels = []
        for img in datal:
            labels.append(img[2])
        if len(labels)!=1:
            labels = np.concatenate(np.array(labels))
        else:
            labels = np.array(labels)

        iter_labels = CustomIterator(labels)
        #if not constant, we have to define an iterator
        if not args.constant_relative_change:
            relative_changes = []
            for img in datal:
                relative_changes.append(relative_tumor_change_dict[img[-1][0]]["relative_change"])
            relative_changes = np.array(relative_changes)
            iter_relative_changes = CustomIterator(relative_changes)
        
        def get_i (t):
            """
            Function to get the target value for the specific sample we are running.
            """
            if args.constant_relative_change:
                if t == int(args.noise_level-2):
                    label =  iter_labels.__next__()
                else: 
                    label = iter_labels.__current__()
                return_label = label + args.relative_change * label
            else:
                if t == int(args.noise_level-2):
                    label = iter_labels.__next__()
                    return_label = label + iter_relative_changes.__next__() * label
                else:
                    label = iter_labels.__current__()
                    return_label =  label + iter_relative_changes.__current__()*label
            #higher noise levels, the target is amplified by 15%
            if t>20 : 
                return_label = return_label + return_label*0.15
            return return_label

        def cond_fn(x, t,  y=None):
            assert y is not None
            with th.enable_grad():
                i = get_i(t)
                x_in = x.detach().requires_grad_(True)
                pred = regressor(x_in, t)
                if t%50 ==0 or t<30:
                    logger.log(f"step {t.detach().cpu().numpy()[0]} pred {pred.detach().cpu().numpy()[0][0]}")
                s_t = i  - pred # -1 shrinks the tumor, 1 makes it grow
            
                a=th.autograd.grad(pred.sum(), x_in)[0]
                
                ##DISABLED -- if additional guidance via a classifier is added 
                # if t>=50 or not args.classifier_path:
                #     return  a, s_t *  a * args.regressor_scale
                # else:
                #     y = th.randint(low=0, high=1, size=(args.batch_size,), device=dist_util.dev() )
                #     logits = classifier(x_in, t)
                #     log_probs = F.log_softmax(logits, dim=-1)
                #     selected = log_probs[range(len(logits)), y.view(-1)]
                #     b=th.autograd.grad(selected.sum(), x_in)[0]
                #     return a , s_t * a * args.regressor_scale + b * args.classifier_scale
                    
                return  a, s_t *  a * args.regressor_scale


        def model_fn(x, t, y=None):
            assert y is not None
            return model(x, t, y if args.class_cond else None)

        logger.log("sampling...")
        all_images = []
        all_labels = []

        #define list to hold samples, anomaly maps and patient identifiers
        guided_diffusion_samples_list = []
        all_samples=[] #samples array
        all_anomalies=[] #anomaly array 
        patient_ids=[] #patient identifiers


        image_count = 0
        for img in datal:
            start_time_generation = time.time()

            model_kwargs = {}
            patient_no = img[4][0].split("'")[0]
            patient_ids.append(patient_no)
        
            if args.class_cond:
                classes = th.randint(
                    low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
                ) #conditioning on the slice class (0-healthy, 1-diseased)
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
                #add the samples images to the samples array
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

            image_count=image_count+1
            if (image_count%100==0) and random_seed==0:
                guided_diffusion_samples_list=[all_samples, all_anomalies, patient_ids]
                with open(f'{saving_dir}/gd_samples_list_{image_count}_rs_{random_seed}.pkl', 'wb') as f:
                    pickle.dump(guided_diffusion_samples_list, f)
            
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]

        guided_diffusion_samples_list=[all_samples, all_anomalies, patient_ids]

        if random_seed==0:
            with open(f'{saving_dir}/gd_samples_list_final_rs{random_seed}.pkl', 'wb') as f:
                pickle.dump(guided_diffusion_samples_list, f)

        logger.log("sampling complete for a generation")
        logger.log(random_seed+1)

        if generation_number != 1:
            prob_maps_img_list.append(all_samples)
            prob_maps_anomaly_list.append(all_anomalies)
            prob_maps_patients_list.append(patient_ids)

        end_time_generation = time.time()
        elapsed_time_generation = end_time_generation - start_time_generation
        print("Elapsed Time (GPU):", elapsed_time_generation)

        #add samples to the all_samples_generations
        all_samples_generations.append(all_samples)
        

    def save_probability_maps(prob_maps_img_list, prob_maps_anomaly_list, prob_maps_patients_list, saving_dir):
        """Robust function to save probability maps for multiple generations"""
        if not prob_maps_img_list:
            return
            
        print("Saving probability maps...")
        
        # Get unique patients
        unique_patients = list(prob_maps_patients_list[0])

        
        if len(unique_patients) == 1:
            # Single patient - save directly
            patient_dir = os.path.join(saving_dir, str(unique_patients[0]))
            os.makedirs(patient_dir, exist_ok=True)
            
            try:
                img_array = np.array([np.array(gen) for gen in prob_maps_img_list])
                anomaly_array = np.array([np.array(gen) for gen in prob_maps_anomaly_list])
            except:
                # Fallback to lists if arrays fail
                img_array = prob_maps_img_list
                anomaly_array = prob_maps_anomaly_list
            print("Img array shape ", img_array.shape)    
            pickle.dump(img_array, open(f"{patient_dir}/generated.pkl", "wb"))
            pickle.dump(anomaly_array, open(f"{patient_dir}/anomalies.pkl", "wb"))
            print(f"Saved for patient {unique_patients[0]}")

        else: 
            # Multiple patients - save each patient separately
            imgs_array = np.array(prob_maps_img_list)      # Shape: [generations, patients, ...]  
            anomalies_array = np.array(prob_maps_anomaly_list)  # Shape: [generations, patients, ...]
            
            for i, patient in enumerate(unique_patients):
                patient_dir = os.path.join(saving_dir, str(patient))
                os.makedirs(patient_dir, exist_ok=True)
                
                # Extract data for patient i across all generations
                patient_imgs = imgs_array[:,i,:]      # All generations for patient i
                patient_anomalies = anomalies_array[:,i,:]

                print("Patient imgs shape ", patient_imgs.shape)
                pickle.dump(patient_imgs, open(f"{patient_dir}/generated.pkl", "wb"))
                pickle.dump(patient_anomalies, open(f"{patient_dir}/anomalies.pkl", "wb"))
                
            print(f"Saved individual data for {len(unique_patients)} patients")

    #saving porbability maps if inference was ran more than once 
    if generation_number != 1:
            save_probability_maps(prob_maps_img_list, prob_maps_anomaly_list, prob_maps_patients_list, saving_dir)
    else:
        print("Saved the final list of samples. ")


    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        regressor_path="",
        regressor_scale=50000,
        noise_level=200,
        dataset='brats',
        partition = False, 
        results_dir = './results',
        partition_number = 0.1, 
        single_contrast = False,
        device='cluster',
        i = 0.1,
        random_seed = 20,
        classifier_path = "",
        classifier_scale = 100,
        constant_relative_change = False,
        relative_change = 0.25,
        relative_tumor_change_dict = '',
        probability_map=False,
        list_patients="",
        test_flag = True,
        run_id = "",
        generation_number = 50, 
        number_in_channels = 4, 
        config = "./config_local/config_inference.yaml"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        defaults.update(config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
