[user]: BMDS-ETH
[repo]: GuidedDiffAnatomicalTumorGrowth 

[issues-shield]: https://img.shields.io/github/issues/BMDS-ETH/GuidedDiffAnatomicalTumorGrowth
[issues-url]:https://github.com/darialaslo/Mechanistic-Learning-with-Guided-Diffusion-Models-to-Predict-Spatio-Temporal-Brain-Tumor-Growth/issues

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Issues][issues-shield]][issues-url]


<div align="center">
<h1 align="center"> Mechanistic Learning with Guided Diffusion Models to Predict Spatio-Temporal Brain Tumor Growth </h1>
Accepted at LMID Workshop at MICCAI, 2025
  <p align="center">
    <a href="https://github.com/darialaslo/Mechanistic-Learning-with-Guided-Diffusion-Models-to-Predict-Spatio-Temporal-Brain-Tumor-Growth/issues">Report Bug</a>
  </p>

   <p align="center">
     <img src="assets/fig_overview_MICCAI.png" alt="Alt text" width="600">
   </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#-project-overview">Project Overview</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#training">Training</a>
      <ul>
        <li><a href="#data-format">Data Format</a></li>
        <li><a href="#diff-training">Diffusion model training</a></li>
        <li><a href="#reg-training">Regressor training</a></li>
      </ul>
    </li>
    <li>
      <a href="#inference">Inference</a>
    </li>
    <li>
      <a href="#mechanistic-modeling">Mechanistic Modeling</a>
    </li>
    <li>
      <a href="#mechanistic-lerning">Mechanistic Learning</a>
    </li>
    <li>
      <a href="#project-organization">Project Organization</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## 📋 Project Overview

Predicting the spatio-temporal progression of brain tumors is essential for guiding clinical decisions in neuro-oncology. We propose a hybrid mechanistic learning framework that combines a mathematical tumor growth model with a guided denoising diffusion implicit model (DDIM) to synthesize anatomically feasible future MRIs from preceding scans. The mechanistic model, formulated as a system of ordinary differential equations, captures temporal tumor dynamics including radiotherapy effects and estimates future tumor burden. These estimates condition a gradient-guided DDIM, enabling image synthesis that aligns with both predicted growth and patient anatomy. We train our model on the BraTS adult and pediatric glioma datasets and evaluate on 60 axial slices of in-house longitudinal pediatric diffuse midline glioma (DMG) cases. Our framework generates realistic follow-up scans based on spatial similarity metrics. It also introduces tumor growth probability maps, which capture both clinically relevant extent and directionality of tumor growth as shown by 95th percentile Hausdorff Distance. The method enables biologically informed image generation in data-limited scenarios, offering generative-space-time predictions that account for mechanistic priors.

#### Code Repository 
This project focuses on the use of computer vision approaches for the generaton of spation-temporal brain tumor growth, with a focus on pediatric data and an integration with methematical modeling. 
We base our approach on a denoising diffusion implicit model (DDIM) and guide the sampling process using the gradient of a separate model.
The repository was developed as an extension of [previous work](https://gitlab.ethz.ch/BMDSlab/publications/oncology/guided-denoising-diffusion-models-for-brain-tumor-mri) (in course of publication) relying on [Wolleb et al.'s implementation](https://gitlab.com/cian.unibas.ch/diffusion-anomaly/-/tree/main/). 

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/darialaslo/Mechanistic-Learning-with-Guided-Diffusion-Models-to-Predict-Spatio-Temporal-Brain-Tumor-Growth.git
    cd Mechanistic-Learning-with-Guided-Diffusion-Models-to-Predict-Spatio-Temporal-Brain-Tumor-Growth
    ```

2. Install the requirements
    ```sh
    pip install -r requirements.txt
    ```
    
    **NOTE**: Critical requirements: Python 3.9+ & PyTorch 1.7+


## Training 

Our guided diffusion network relies on DDPM and a regressor (U-Net encoder with time-step embeddings) trained independently. Details on the training procedure and default arguments are provided below. Generally, configuration files available in the config folder can be used for the training and testing of models, they contain the default values for the arguments and details on their use. Config files are provided for the main training, testing and inference scripts.

### Data format

The BraTS dataset was used for training. Multiparametric axial 2D slices were used for training, testing and validation, which must be organized as follows (see data folder for a few examples):

    /data/
      training/
        pid_slice
          dataset_train_pid_seg_slice_w.nii
          dataset_train_pid_t1n_slice_w.nii
          dataset_train_pid_t1c_slice_w.nii
          dataset_train_pid_t2w_slice_w.nii
          dataset_train_pid_t2f_slice_w.nii
        pid_slice
          ...
      testing/
        pid_slice
            dataset_train_pid_<t1n/t1c/t2w/t2f>_slice_w.nii
        ...
      validation/
        pid_slice
            dataset_train_pid_<t1n/t1c/t2w/t2f>_slice_w.nii
        ...
      test_labels/
          pid_slice-label.nii.gz (test)
          pid_slice-label.nii.gz (validation)
          ...


### Diffusion model training


The 'scripts/image_train.py' script needs to be run for training the regression model, while setting the following default and specific parameters which can be found in the ```config/config_diff_train.yaml``` file. A minimal working version including the default settings is provided there. This should run with the reduced dataset provided for the testing of the codebase. 
For specific configurations a path to another config file can be specified as a command line argument ```--config ./config/new_config.yaml```. Specifically, the default parameters are:

```
--lr 1e-4 --batch_size 10 --dataset brats --image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False

```

Additionally, specific arguments depending on the data and use case must be provided:
* --data_dir : path to the master directory containing your data (needs to be structured in the following directories: training, validation, testing, test_labels)
* --resume_checkpoint : path to partly trained model, if continuing training is desired
* others - additional arguments can be found in the config file

**Validation**: The script scripts/diffusion_validation.py can be used to compute the overall MSE for the training set and the validation set for a given state of the model. 

### Regressor training 

The 'scripts/regressor_train.py' script needs to be run for training the regression model, while setting the following default and specific parameters which can be found in the ```config/config_reg_train.yaml``` file. A minimal working version including the default settings is provided there. This should run with the reduced dataset provided for the testing of the codebase. 
For specific configurations a path to another config file can be specified as a command line argument ```--config ./config/new_config.yaml```. Specifically, the default parameters are:

```
--lr 1e-4 --batch_size 10 --dataset brats --image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --regressor_attention_resolutions 32,16,8 --regressor_depth 4 --regressor_width 32 --regressor_pool attention --regressor_resblock_updown True --regressor_use_scale_shift_norm True
```

Additionally, specific arguments depending on the data and use case must be provided:
* --data_dir : path to the master directory containing your data (needs to be structured in the following directories: training, testing, test_labels)(multiple directories can be specified if separated by commas)
* --resume_checkpoint : path to partly trained model, if continuing training is desired
* others - additional arguments can be found in the config file

**Validation**: The script can be instructed to perform validation every given number of iterations and store results. 

**Testing**: The script scripts/regression_test.py can be used to apply the selected regression model to the test set and calculate MSE for the predicted values. 

## Inference 

To generate samples using the guided diffusion network, the gd_regressor_sample_known.py needs to be run. The paths of the already trained diffusion model and regression model need to be provided. The specific parameters can be found in the ```config/config_gd.yaml``` file. The generation can be guided towards a fixed relative change, by setting ```--constant_relative_change True``` and specifying the exact value with ```--relative_change xxx```. If specific relative changes are expected for the different samples, a dictionary (saved as pickle) is defined, specifying the sample as key and the target image (if available: longitudinal samples) and the relative change (an example file is provided for the dummy validation set in ```data/targets_dict.pkl```). 

**NOTES**:
* The name for the model_path or regressor_path should be in the following format: modelxxxxxx.pt, where xxxxxx is the the number of iterations the training ran for. 


## Mechanistic modeling

## Mechanistic learning


## Project Organization TODO

--------
    ├── data                  
    │   ├── training             <- The training data, each sample one directory containing the relevant files (T1N, T1C, T2W, T2F, SEG).
        ├── validation           <- The validation data, each sample one directory containing the relevant files (T1N, T1C, T2W, T2F).
        ├── testing              <- The test data, each sample one directory containing the relevant files (T1N, T1C, T2W, T2F).
    │   ├── test_labels          <- The segmentation files for all testing and validation samples.
    ├── assets               <- Documentation assets, manuals, and all other explanatory materials.
    │
    │
    ├── src_guided_denoising                <- Source code for the denoising diffusion and guided diffusion inference framework 
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── guided_diffusion         <- Guided diffusion scripts.
    │   └── scripts                  <- Training and inference scripts for the guided diffusion framework.
    │      ├── config                <- Config files for running scripts, including default arguments. 
    │      ├── .py
    │      └── .py
    │   
    ├── src_mechanistic_model         <- Scripts for the mechanistic modeling of tumor growth. #TODO Efthymis
    │       ├── ... 
    │       ├── ....py
    │       └── ....py
    ├── utils       <- Related utils for training and downstream analyses 
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    └── requirements.txt   <- The requirements file for reproducing the analysis environment generated with `pip freeze > requirements.txt`

--------

<!-- LICENSE -->
## License

Distributed under the Apache License Version 2.0, License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [BraTS Dataset Adult](synapse.org/Synapse:syn51156910) We gratefully acknowledge the BraTS dataset (DOI: 10.1109/TMI.2014.2377694) provided through the BraTS Challenge as part of MICCAI.  This comprehensive multi-center multi-parametric dataset with tumor segmentations has been instrumental in training and validating our models.
* [BraTS Dataset PED](synapse.org/Synapse:syn51156910) We gratefully acknowledge the SPIDER dataset (DOI: 10.59275/j.melba.2025-f6fg) provided through the BraTS Challenge as part of MICCAI.   This comprehensive multi-center multi-parametric **pediatric** dataset with tumor segmentations has been instrumental in training and validating our models.
* [Wolleb et al.](10.1007/978-3-031-16452-1_4) Our implementation builds on the repository developed by Wolleb et al. (https://gitlab.com/cian.unibas.ch/diffusion-anomaly).

<!-- CONTACT -->
## Contact

Daria Laslo - PhD Student at ETH Zürich Biomedical Data Science Lab 
[Email Me](daria.laslo@hest.ethz.ch)  
[Github Profile](https://github.com/darialaslo)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

