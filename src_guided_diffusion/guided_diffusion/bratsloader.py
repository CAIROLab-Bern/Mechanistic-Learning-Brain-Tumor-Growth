import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage
from sklearn.model_selection import train_test_split
import random
import random
import pickle

    #in intialisation the database is defined as an array of dictionaries (datapoint here)
    #each dictionary contains the 4 contrasts + segmentation for that patient 

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, partition=False, device = 'cluster', validation_flag = False, partition_number = 0.1, single_contrast = False, remove_small = False, regression = False, list_patients = "", type_slices="all" ):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
       
        directory = directory.split(',')

        if len(directory) != 1:
            print("More than one directory given, will load all of them")
            print("Attention: this option is valid only for the training set, not for the test set")
            self.directory = [os.path.expanduser(d) for d in directory]

        else:
            self.directory = [os.path.expanduser(directory[0])]
        print("Loading from: ", self.directory)

        self.device=device
        self.directory=directory
        self.test_flag=test_flag
        self.validation_flag = validation_flag
        self.partition=partition
        self.single_contrast = single_contrast
        self.regression = regression
        random.seed(12) #set random seed

        #need to account for different sequences if we have train/test samples
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        if single_contrast:
            self.seqtypes = ['t2f'] #if single constrast only select t2-flair, can be changed to the choice of sequence
        print(test_flag)
        print(validation_flag)
        if not test_flag and not validation_flag:
            self.seqtypes.append('seg') #expect segmentaiton in the training folder

        self.seqtypes_set = set(self.seqtypes)
        print(self.seqtypes_set)
        self.database = []


        #subsetting directory to specific samples or a fraction of the input directory
        if partition:
            if list_patients!="":
                print("I'm setting the partition to the list of patients.")
                test_partition_2 = list_patients
            else:
                testing_subdirectories = next(os.walk(self.directory[0]))[1]
                test_partition_1 , test_partition_2 = train_test_split(testing_subdirectories, test_size=partition_number, random_state=12)
                
        for directory in self.directory:
            skipped = 0
            for root, dirs, files in os.walk(directory):
                if not dirs and (not partition or partition and root.split("/")[-1] in test_partition_2):      
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        # if f.find("original")!=-1: #condition for loading inference samples or from directory containing other files
                        #     continue
                        seqtype = f.split('_')[3]
                        if seqtype in self.seqtypes_set:
                            datapoint[seqtype] = os.path.join(root, f)
                    try:
                        assert set(datapoint.keys()) == self.seqtypes_set, \
                        f'datapoint is incomplete, keys are {datapoint.keys()}, {f}'
                    except:
                        print(f'datapoint is incomplete, keys are {datapoint.keys()}, {f}') 
                        print("Skipping this data point.")
                        skipped+=1
                        continue
                    #Choose which class to load: diseased or healthy
                    if type_slices != "all":
                        seg_nib = nibabel.load(datapoint["seg"])
                        seg = seg_nib.get_fdata()
                        if type_slices=="healthy" and np.sum(seg)>0:
                            print("Skipping datapoint, it is diseased and I want only healthy slices")
                            skipped+=1
                            continue
                        elif type_slices=="diseased" and np.sum(seg)<=0:
                            print("Skipping datapoint, it is healthy and I want only diseased slices")
                            skipped+=1
                            continue
                    self.database.append(datapoint)
        print("The length of the database is ", len(self.database))

    def __getitem__(self, x):
        filedict = self.database[x]
        number = filedict['t2f'].split('/')[-2]

        # Load and preprocess each sequence type
        out = []
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype]).get_fdata()
            if nib_img.shape != (240, 240):
                nib_img = nib_img[8:-8, 8:-8]
            out.append(torch.tensor(nib_img))
        out = torch.stack(out)

        # Setup image tensor
        channels = 1 if self.single_contrast else 4
        image = torch.zeros(channels, 256, 256)
        image[:, 8:-8, 8:-8] = out if self.test_flag else out[:-1]

        # Initialize output dict
        out_dict = {}

        # Helper function to compute weak label
        def compute_weak_label(seg_tensor):
            if seg_tensor.max() > 0:
                count_label = np.count_nonzero(seg_tensor)
                ref_channel = 0 if self.single_contrast else 3
                count_brain = np.count_nonzero(out[ref_channel])
                return np.float32(count_label / count_brain) if count_brain != 0 else 0
            return 0 if self.regression else int(seg_tensor.max() > 0)

        if self.test_flag:
            src_directory = filedict['t2f'].split('/testing')[0] if 'testing' in filedict['t2f'] else filedict['t2f'].split('/validation')[0]
            seg_path = f'{src_directory}/test_labels/{number}-label.nii.gz'
            seg = nibabel.load(seg_path).get_fdata()
            label = torch.zeros(1, 256, 256)
            label[:, 8:-8, 8:-8] = torch.tensor(seg)[None, ...]
            label[label > 0] = 1
            weak_label = compute_weak_label(seg)
            out_dict['y'] = weak_label

        elif self.validation_flag:
            label = out[-1][None, ...]
            weak_label = 1
            out_dict['y'] = 1

        else:  # Training set
            label = torch.zeros(1, 256, 256)
            label_slice = out[-1]
            label[:, 8:-8, 8:-8] = label_slice[None, ...]
            label[label > 0] = 1

            if self.regression:
                weak_label = compute_weak_label(label_slice)
            else:
                weak_label = int(label.max() > 0)

            out_dict['y'] = weak_label

        return image, out_dict, weak_label, label, number


    def __len__(self):
        return len(self.database)


    def get_slice(self,x):

        filedict = self.database[x]
        if self.device == 'laptop' or self.device =="euler" :
            number=filedict['t2f'].split('/')[8] #modified to two due to the different organization number=filedict['t1'].split('/')[4]
        else:
            number=filedict['flair'].split('/')[7] #modified to two due to the different organization number=filedict['t1'].split('/')[4]
        return number.split("_")[-1]

    def get_id(self,x):
        filedict = self.database[x]
        number=filedict['t2f'].split('/')[-2] #modified to two due to the different organization number=filedict['t1'].split('/')[4]
        return number
