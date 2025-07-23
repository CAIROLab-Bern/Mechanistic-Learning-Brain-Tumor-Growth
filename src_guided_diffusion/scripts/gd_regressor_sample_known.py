"""
Generate images with enalrged tumors starting from a baseline image. 
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
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from utils.script_util import (
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

    #create folder to save figures
    if not os.path.isdir(f"{args.results_dir}/figures"):
        os.mkdir(f"{args.results_dir}/figures")
    saving_dir = f"{Path(args.data_dir).parent}/{args.run_id}"
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    print(f"Saving samples to {saving_dir}.")
    

    #if we don't have constant change we have a dictionary specifying the change 
    if not args.constant_relative_change:
        print("Tumor change is defined within a dictionary")
        assert args.relative_tumor_change_dict!='', "Constant relative change is not enabled. Enable constant relative tumor change or provide a dictionary specifying change for every sample!"
        #read the specific relative change from the dictionary
        print(args.relative_tumor_change_dict)
        
        relative_tumor_change_dict = pickle.load(open(args.relative_tumor_change_dict, "rb"))
        print(list(relative_tumor_change_dict.keys()))
        #set the list of patients to the patients in that dictionary
        # args.list_patients = list(relative_tumor_change_dict.keys())
        print("The list of patients is: ", args.list_patients)

    if args.dataset=='brats' and args.data_dir.find("DIPG")==-1:
      ds = BRATSDataset(args.data_dir, test_flag=args.test_flag, partition=args.partition, partition_number = args.partition_number, single_contrast = args.single_contrast, remove_small=args.remove_small, device=args.device, regression = True, list_patients=args.list_patients)
      datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    elif args.dataset=='chexpert':
     data = load_data(
         data_dir=args.data_dir,
         batch_size=1,
         image_size=args.image_size,
         class_cond=True,
     )
     datal = iter(data)
    else:
        ds = DIPGDataset(args.data_dir, test_flag=False, partition=args.partition, partition_number = args.partition_number, single_contrast = args.single_contrast, remove_small=args.remove_small, device=args.device, list_patients = args.relative_tumor_change_dict)
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
        generation_number = 50
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

    all_samples_generations = []
    gradients_list = []
    noisy_version_list = []

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

        regressor.to(dist_util.dev())
        if args.regressor_use_fp16:
            regressor.convert_to_fp16()
        regressor.eval()

        #extracting the labels, we will use this to guide the tumors to a size relative to the starting one
        
        labels = []
        for img in datal:
            labels.append(img[2])
        if len(labels)!=1:
            print(" The length of labels is ", len(labels))
            print("The labels are ", labels)
            labels = np.concatenate(np.array(labels))
            print("The labels are ", labels)
        else:
            labels = np.array(labels)
        labels = labels[labels!=0]

        iter_labels = CustomIterator(labels)
        if not args.constant_relative_change:

            relative_changes = []
            for img in datal:
                # print(img[-1])
                # print(img[-1][0])
                relative_changes.append(relative_tumor_change_dict[img[-1][0]]["relative_change"])

            print(relative_changes)
            relative_changes = np.array(relative_changes)

            iter_relative_changes = CustomIterator(relative_changes)
        
        def get_i (t):

            if args.constant_relative_change:
                if t == int(args.noise_level-2):
                    print(t)
                    label =  iter_labels.__next__()
                    
                else: 
                    label = iter_labels.__current__()
                # print("The label is ", label)
                return_label = label + args.relative_change * label
            else:
                if t == int(args.noise_level-2):
                    print(t)
                    label = iter_labels.__next__()
                    return_label = label + iter_relative_changes.__next__() * label
                else:
                    label = iter_labels.__current__()
                    return_label =  label + iter_relative_changes.__current__()*label
            print("The label is ", label)
            print("The return label is ", return_label)
            return return_label

        def cond_fn(x, t,  y=None):
            assert y is not None
            with th.enable_grad():
                i = get_i(t)
                x_in = x.detach().requires_grad_(True)
                pred = regressor(x_in, t)
                if t%50 ==0:
                    logger.log(f"step {t.detach().cpu().numpy()[0]} pred {pred.detach().cpu().numpy()[0][0]}")
                
                s_t = i - pred
                print("y", y)
                # s_t = 1 # -1 shrinks the tumor, 1 makes it grow
                a=th.autograd.grad(pred.sum(), x_in)[0]
                # if t%50==0 and (random_seed == 0 or random_seed == 10):
                if  (random_seed == 0 or random_seed == 10):
                        gradients_list.append(a.cpu().numpy()[0,3,...])
                        noisy_version_list.append(x_in.detach().cpu().numpy()[0,3,...])
                        # plt.imshow(a.cpu().numpy()[0,3,...], cmap='gray')
                        # plt.colorbar()
                        # plt.savefig(f"{args.results_dir}/figures/flair_grad_iteration{random_seed}_step{t.cpu().numpy()[0]}.png")    
                        # plt.close()
                        # plt.imshow(x_in.detach().cpu().numpy()[0,3,...], cmap='gray')
                        # plt.colorbar()
                        # plt.savefig(f"{args.results_dir}/figures/flair_image_iteration{random_seed}_step{t.cpu().numpy()[0]}.png")    
                        # plt.close()
                        # # plt.imshow(a.cpu().numpy()[0,1,...], cmap='gray')
                        # # plt.colorbar()
                        # # plt.savefig(f"{args.results_dir}/figures/t1ce_grad_iteration{random_seed}_step{t.cpu().numpy()[0]}.png")    
                        # plt.close()
                #  
                # print(a)

                if t>=50 or not args.classifier_path:

                    return  a, s_t *  a * args.regressor_scale 

                else:
                    print("I have less than 50 steps left")
                    y = th.randint(low=0, high=1, size=(args.batch_size,), device=dist_util.dev() )
                    logits = classifier(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    b=th.autograd.grad(selected.sum(), x_in)[0]
                     
                    # if t%100==0:
                    #     plt.imshow(b.cpu().numpy()[0,3,...], cmap='gray')
                    #     plt.savefig(f"{args.results_dir}/figures/grad_iteration{random_seed}_step{t}.png")
                    # print(b)

                    return a , s_t * a * args.regressor_scale + b * args.classifier_scale
                    
                
                return  a, s_t *  a * args.regressor_scale
            
                # x_in = x.detach().requires_grad_(True)
                # logits = classifier(x_in, t)
                # log_probs = F.log_softmax(logits, dim=-1)
                # selected = log_probs[range(len(logits)), y.view(-1)]
                # a=th.autograd.grad(selected.sum(), x_in)[0]
                # return  a, a * args.classifier_scale

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
        #   img = next(data)  # should return an image from the dataloader "data"
            print('img', img[0].shape, img[1])
            if args.dataset=='brats':
                Labelmask = th.where(img[3] > 0, 1, 0)
                number=img[4][0]
                if img[2]==0:
                    continue    #take only diseased images as input

                #add the patient number to patient ids
                patient_no = img[4][0].split("'")[0]
                patient_ids.append(patient_no)

            else:
                 
                print('img1', img[1])
                number=img[1]["path"]
                print('number', number)
                print(img[0].shape)
                patient_no = 'lalal'
                patient_ids.append(patient_no)


            # model_kwargs["i"] = 0.1 #hardcoded for now
            # print('i', model_kwargs["i"])
            if args.class_cond:
                classes = th.randint(
                    low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
                print('y', model_kwargs["y"]) 
                
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            print('samplefn', sample_fn)
            # start = th.cuda.Event(enable_timing=True)
            # end = th.cuda.Event(enable_timing=True)
            # start.record()
            sample, x_noisy, org = sample_fn(
                model_fn,
                (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise_level=args.noise_level
            )
            # end.record()
            # th.cuda.synchronize()
            # th.cuda.current_stream().synchronize()
            # print('time for 1000', start.elapsed_time(end))

            if args.dataset=='brats':
                 
                if not args.single_contrast:
                     
                     
                     
                    all_samples.append(sample.cpu().numpy().reshape((4,256,256)))
                else:
                    all_samples.append(sample.cpu().numpy().reshape((1,256,256)))
                difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
                 

                #add the samples images to the samples array
                
                all_anomalies.append(difftot.cpu().numpy())
            
            elif args.dataset=='chexpert':
                 
                diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
                diff=np.array(diff.cpu())
                 
                all_samples.append(sample.cpu().numpy().reshape((1,256,256)))
                all_anomalies.append(diff)

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
            if (image_count%100==0):
                src_samples_list=[all_samples, all_anomalies, patient_ids]
                with open(f'{args.results_dir}/gd_samples_list_{image_count}_rs_{random_seed}.pkl', 'wb') as f:
                    pickle.dump(src_samples_list, f)
            
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]

        src_samples_list=[all_samples, all_anomalies, patient_ids]

        with open(f'{args.results_dir}/gd_samples_list_final_{random_seed}.pkl', 'wb') as f:
            pickle.dump(src_samples_list, f)
        #save gradients and noisy versions
        with open(f'{args.results_dir}/gradients_{random_seed}.pkl', 'wb') as f:
            pickle.dump(gradients_list, f)
        with open(f'{args.results_dir}/noisy_{random_seed}.pkl', 'wb') as f:
            pickle.dump(noisy_version_list, f)
        

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
        

    if generation_number != 1:
            print("I am saving the probability maps")
            print("The shape of the all samples array is ", np.array(all_samples_generations).shape)
            if np.array(all_samples_generations).shape[1]==1:
                prob_maps_img_array = np.array(prob_maps_img_list).reshape((generation_number, np.array(all_samples_generations).shape[2], np.array(all_samples).shape[3], np.array(all_samples).shape[3]))
                prob_maps_anomaly_array = np.array(prob_maps_anomaly_list).reshape(( generation_number, np.array(all_anomalies).shape[2], np.array(all_anomalies).shape[2]))
            else:

                prob_maps_img_array = np.array(prob_maps_img_list).reshape((np.array(all_samples_generations).shape[0], generation_number, np.array(all_samples).shape[1], np.array(all_samples).shape[2], np.array(all_anomalies).shape[2]))
                prob_maps_anomaly_array = np.array(prob_maps_anomaly_list).reshape((np.array(all_anomalies).shape[0], generation_number, np.array(all_anomalies).shape[1], np.array(all_anomalies).shape[2], np.array(all_anomalies).shape[2]))
            print("The shape of the prob_maps_img_array is ", prob_maps_img_array.shape)

            if not os.path.isdir(f"{saving_dir}/{prob_maps_patients_list[0][0]}"):
                os.mkdir(f"{saving_dir}/{prob_maps_patients_list[0][0]}")

            ###IMPT: only works if I have one patient running
            pickle.dump(prob_maps_img_array, open(f"{saving_dir}/{prob_maps_patients_list[0][0]}/generated.pkl", "wb"))
            pickle.dump(prob_maps_anomaly_array, open(f"{saving_dir}/{prob_maps_patients_list[0][0]}/anomalies.pkl", "wb"))

            #OOOLLLDDD
            #refactoring the list such that we keep the same structure list = [generated_imgs, anomalies, patients], each of these being a list of the length equal to the number of generations
            # prob_maps_saving = [prob_maps_img_array,prob_maps_anomaly_array, prob_maps_patients_list]

            # list_0 = [np.array(prob_maps_saving[0])[i,::] for i in range(generation_number)]
            # list_1 = [np.array(prob_maps_saving[1])[i,::] for i in range(generation_number)]
            # list_2 = [np.array(prob_maps_saving[2])[i,::] for i in range(generation_number)]

            # prob_maps_saving = np.array([list_0, list_1, list_2])

            # with open(f'{args.results_dir}/PM_gen.pkl', 'wb') as f:
            #    pickle.dump(prob_maps_saving, f)
            # print("Saved the Probability maps")


            # #delete the singular files 
            # for root, dirs, files in os.walk(args.results_dir):
            #             for f in files:
            #                 if f.find("samples_list")!=-1:
            #                     os.remove(root+"/"+f)
            # print("Deleted all singular files ")
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
        regressor_scale=100,
        noise_level=500,
        dataset='brats',
        partition = False, 
        results_dir = './results',
        partition_number = 0.1, 
        single_contrast = False,
        remove_small=False,
        device='cluster',
        i = 0.1,
        relative_change = 0.25,
        random_seed = 20,
        classifier_path = "",
        classifier_scale = 100,
        constant_relative_change = True,
        relative_tumor_change_dict = '',
        probability_map=False,
        list_patients="",
        test_flag = True,
        run_id = ""

    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

