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

class GDDataset(torch.utils.data.Dataset):
    def __init__(self, file='', directory='', device ='cluster', test_flag=False, single_contrast=False, regression = False, source = 'file'):
        '''
        
        '''
        super().__init__()

        self.source = source
        self.test_flag=test_flag
        self.single_contrast = single_contrast
        self.regression = regression
       
        if self.source=='file':

            file = file.split(',')

            if len(file) != 1:
                print("More than one file given, will load all of them")
                print("Attention: this option is valid only for the training set, not for the test set")
                # assert not test_flag, "More than one file given for the test set, this is not allowed!!"
                self.file = [os.path.expanduser(d) for d in file]

            else:
                self.file = [os.path.expanduser(file[0])]
            self.device=device

            #load the pickle file 

            images = []
            patient_numbers = []
        
            for f in self.file:
                gd_samples_list = pickle.load(open(f, 'rb'))
                # print(gd_samples_list[2])
                # print(np.concatenate(gd_samples_list[2]))
                if isinstance(gd_samples_list, list) or (not isinstance(gd_samples_list, list) and gd_samples_list.shape[0]==3):
                    if single_contrast:
                        print(np.array(gd_samples_list[0][0]).shape)
                        if len(np.array(gd_samples_list[0]).shape)==1:
                            squeezed  = np.array([np.squeeze(i) for i in gd_samples_list[0]])
                            print(np.array(squeezed).shape)
                            images.append(squeezed[:,3,:,:].reshape((squeezed.shape[0],1,squeezed.shape[-1],squeezed.shape[-1])))
                        elif np.array(gd_samples_list[0][0]).shape!=(4,256,256):
                            print("In the stupid elif")
                            for i in range(np.array(gd_samples_list[0][0]).shape[0]):
                                for j in range(np.array(gd_samples_list[0]).shape[1]):
                                    print(np.array(gd_samples_list[0]).shape)
                                    print(np.array(gd_samples_list[0])[i,j].shape)
                                    images.append(np.array(gd_samples_list[0])[i,j][3,:,:].reshape((1,256,256)))
                                    patient_numbers.append(np.array(gd_samples_list[2])[i,j])
                        elif np.array(gd_samples_list[0][0]).shape==(4,256,256) and len(np.array(gd_samples_list[0]).shape)==4:
                            print("In the right elif ")
                            for i in range(np.array(gd_samples_list[0]).shape[0]):
                                images.append(np.array(gd_samples_list[0])[i][3,:,:].reshape((1,256,256)))
                                patient_numbers.append(np.array(gd_samples_list[2])[i])
                        else:
                            print(np.array(gd_samples_list[0]).shape)
                            images.append(gd_samples_list[0][3,:,:])
                            
                    else:
                        print(np.array(gd_samples_list[0]).shape)
                        images.append(gd_samples_list[0])
                        
                        if np.array(images).shape[0]==1:
                            images = np.array(images)
                            images = images.reshape((images.shape[1], images.shape[2], images.shape[3], images.shape[4]))
                            patient_numbers = gd_samples_list[2]
                else:
                    images.append(gd_samples_list[:,0])
                    patient_numbers.append(gd_samples_list[2])
            print("Images shape ", np.array(images).shape)
            # self.images = np.concatenate(images)
            self.images = np.array(images)
            print(f"I have {len(self.images)} samples.")
            print("The shape of the images is: ", self.images.shape)
            # self.patient_numbers = np.concatenate(patient_numbers)
            print(gd_samples_list[2])
            print(type(gd_samples_list[2]))
            print(len(gd_samples_list[2]))
            self.patient_numbers = patient_numbers
            print("These are my patients ", self.patient_numbers)
        else:
            self.source='directory'
            self.database = []

            if test_flag :
                self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
                if single_contrast:
                    self.seqtypes = [ 't2f']
            else:
                self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
                if single_contrast:
                    self.seqtypes = [ 't2f', 'seg']
            self.seqtypes_set = set(self.seqtypes)
            print(self.seqtypes_set)

            patient_numbers = []
            for root, dirs, files in os.walk(directory):
                # if there are no subdirs, we have data
                #we also check if we have a partition and we only load the corresponding directories 
                #we also check if we want to remove small tumors and we only load the corresponding directories (removing specific slices)
                if not dirs :
                    patient_numbers.append(root.split("/")[-1])
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        if directory.find("GD")!=-1:
                            seqtype=f.split('_')[2].split(".")[0]
                        else:
                            seqtype=f.split('_')[3].split(".")[0]
                        #if we are using the BraTS23 pediatric dataset, we need to account for the training data that also contains the segmentation
                        #for the diffusion training, no data with segmentation is needed
                        if seqtype != "anomaly":
                            if single_contrast :
                                if seqtype in self.seqtypes_set :
                                    datapoint[seqtype] = os.path.join(root, f)
                            else:
                                datapoint[seqtype] = os.path.join(root, f)
                    assert set(datapoint.keys()) == self.seqtypes_set, \
                        f'datapoint is incomplete, keys are {datapoint.keys()}, {f}'
                    self.database.append(datapoint)
            self.patient_numbers = patient_numbers
            print(len(self.database))




    def __getitem__(self, x):

        if self.source == 'file':
            out = self.images[x]
            out_dict = {}
            label = out[-1, ...][None, ...]
            weak_label = 1
            out_dict["y"] = 1
            image = out
            # assert image.shape==(4,256,256)
            number = self.patient_numbers[x]




        else:
            out = []
            filedict = self.database[x]
            for seqtype in self.seqtypes:
                #need to account for the different contrasts in case we have the BraTS23 pediatric dataset

                number=filedict['t2f'].split('/')[7]
                nib_img = nibabel.load(filedict[seqtype])  # the shape of each image extracted here is 240 x 240
                if not self.single_contrast or not self.test_flag: 
                    out.append(torch.tensor(nib_img.get_fdata().reshape((256,256)), dtype=torch.float))
                else:
                    #depending on whether we have a folder with generated images or original240x240 unsegmented  
                    try:
                        out = torch.tensor(np.float32(nib_img.get_fdata()).reshape((256,256)), dtype=torch.float)
                    except:

                        out = np.zeros((256, 256))
                        out[8:-8,8:-8] = nib_img.get_fdata().reshape((240,240))

                        out = torch.tensor(np.float32(out).reshape((1,256,256)), dtype=torch.float)
            if not self.single_contrast or not self.test_flag:  
                out = torch.stack(out)
            print(out.shape)
            out_dict = {}


            label = torch.zeros(1, 256, 256)
            if self.single_contrast:
                image = torch.zeros(1, 256, 256)
                if not self.test_flag:
                    img_t2f = out[0,...]
                else:
                    img_t2f = out
                
            else:
                image = torch.zeros(4, 256, 256)
                img_t2f = out[3,...]

            if self.test_flag:
                    image = out
            else:
                image=out[:-1,...]		#pad to a size of (256,256)
            label= out[-1, ...][None, ...]
            print("This is the image ")
            print(img_t2f.shape)
            if self.regression:
                seg = out [-1, ...]
                assert seg.shape == (256,256)
                brain_mask = np.zeros((256,256))
                # print(img_t2f[0:8,:].flatten().shape)
                # print(np.max(np.array(img_t2f[0:8,:].flatten())))
                assert img_t2f.shape == (256,256)
                max_value = np.max(np.array(img_t2f[0:8,:].flatten()))
                brain_mask[img_t2f>max_value]=1
                print("The seg mask is ",np.count_nonzero(seg) )

                if  np.count_nonzero(brain_mask)!=0:
                    label = np.count_nonzero(seg) / np.count_nonzero(brain_mask)
                else:
                    label = 0
                print("The relative size is ", label)

                            

            weak_label=1
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        if self.source=="file":
            return len(self.images)
        else:
            return len(self.database)


    def get_data_by_patient(self, patient_no):
   

        idx_patient = np.where(np.array(self.patient_numbers)==patient_no)[0]
        print(idx_patient)
        print(patient_no)
        print(self.patient_numbers)
        print(np.where(self.patient_numbers==patient_no))
        if self.source == "file":
            print(idx_patient)
            print(self.images[ idx_patient ])
            out = self.images[ idx_patient ].reshape((1,256,256))
            out_dict = {}
            label = out[-1, ...][None, ...]
            weak_label = 1
            out_dict["y"] = 1
            image = out
        else:
            return self.__getitem__(idx_patient)

        if self.single_contrast:
            assert image.shape == (1,256,256)
        else:
            assert image.shape==(4,256,256)
        
        return (image, out_dict, weak_label, label, patient_no )

    def get_patient_numbers (self):
        return self.patient_numbers
