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

'''
This loader is meant to create a dataset that combines real and generated images.

'''

class JOINTDataset(torch.utils.data.Dataset):
    def __init__(self, directory_real, directory_generated, device = 'cluster', set_type = "train"):
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

        directory_real = directory_real.split(',')
        directory_generated = directory_generated.split(',')

        if len(directory_real)!=1:
            self.directory_real = [os.path.expanduser(d) for d in directory_real]
        else:
            self.directory_real = [os.path.expanduser(directory_real[0])]
       
        if len(directory_generated)!=1:
            self.directory_generated = [os.path.expanduser(d) for d in directory_generated]
        else: 
            self.directory_generated = [os.path.expanduser(directory_generated[0])]

        self.device=device
        self.set = set_type


        self.seqtypes = ['t1n', 't1c', 't2w', 't2f']


        self.seqtypes_set = set(self.seqtypes)
        self.database = []
 
        #split the data 
        #we wanted the same slices from real and generated in the training set 
        subdirs=[]
        train_subdirs = []
        test_subdirs =[]
        for dir_gen in self.directory_generated:
            print(dir_gen)
            subdirs_current = list(next(os.walk(dir_gen))[1])
            print(subdirs_current)
            train_subdirs_current, test_subdirs_current = train_test_split(subdirs_current, test_size=0.1, random_state=12)
            subdirs = subdirs + subdirs_current
            train_subdirs = train_subdirs + train_subdirs_current
            test_subdirs = test_subdirs + test_subdirs_current



        for directory in self.directory_real+self.directory_generated:
            print(directory)
            
            for root, dirs, files in os.walk(directory):
                # if there are no subdirs, we have data
                #we also check if we have a partition and we only load the corresponding directories 
                #we also check if we want to remove small tumors and we only load the corresponding directories (removing specific slices)

                if not dirs and ((self.set=="train" and root.split("/")[-1] in train_subdirs) or (self.set=="test" and root.split("/")[-1] in test_subdirs)):
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        if directory in self.directory_real:
                            seqtype = f.split('_')[3]
                        else:
                            seqtype = f.split('_')[2].split(".")[0]
                        if (seqtype !="seg"):
                            datapoint[seqtype] = os.path.join(root, f)

                    # assert set(datapoint.keys()) == self.seqtypes_set, \
                    #     f'datapoint is incomplete, keys are {datapoint.keys()}, {f}'
                    self.database.append(datapoint)
        print(self.database[:10])
        random.seed(12)
        random.shuffle(self.database)
        print(self.database[:10])
       

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        # print(filedict)
        source_dir = filedict['t2f'].split("/")[-3]
        number = filedict['t2f'].split('/')[7]

        if source_dir == self.directory_real[0].split("/")[-1]:
            real = True
        else:
            real = False
        # print(source_dir)
        # print(real)
            
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])  # the shape of each image extracted here is 240 x 240
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}

        image = torch.zeros(4, 256, 256)

        if real:
            image[:, 8:-8, 8:-8] = out
            weak_label = 1
        else:
            image[:, :, :] = out
            weak_label = 0 
        #we don't really have a pixel by pixel label
        label = torch.zeros(1, 256, 256)
        out_dict["y"] = weak_label


        return (image, out_dict, weak_label, label, number)

    def __len__(self):
        return len(self.database)



