import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        #files = list(pathlib.Path(root).iterdir())
        self.examples = []
        #self.acc_factor = acc_factor 
        #self.dataset_type = dataset_type
        #self.key_img = 'img_volus_{}'.format(self.acc_factor)
        #self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #acc_factors = ['3_3x','4x','5x']
        #print("acc_factors in slice data: ", acc_factors)
        for dataset_type in dataset_types:
            dataroot = os.path.join(root, dataset_type)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot, mask_type,train_or_valid)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor))).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            #acc_factor = float(acc_factor[:-1].replace("_","."))
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice,acc_factor,mask_type,dataset_type = self.examples[i] 
        # Print statements 
        #print (fname,slice)
        #print ("acc_factor: ",acc_factor)
    
        with h5py.File(fname, 'r') as data:
            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)

            input_img  = data[key_img][:,:,slice]
            #print(key_img)
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            #target = data['volfs'][:,:,slice]

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            #acc_val = torch.Tensor([float(acc_factor[:-1].replace("_","."))])
            acc_val = float(acc_factor[:-1].replace("_","."))
            mask_val = 0 if mask_type=='cartesian' else 1
            dataset_val = 0 if dataset_type=='mrbrain_t1' else 1
            gamma_input = np.array([acc_val, mask_val,dataset_val])
            #print(torch.from_numpy(gamma_input).shape)


            #print (input_img.dtype,input_kspace.dtype,target.dtype,gamma_input.dtype)

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(gamma_input), acc_factor, mask_type,dataset_type

            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,acc_factor,dataset_type,mask_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        #self.acc_factor = acc_factor
        #self.dataset_type = dataset_type

        #self.key_img = 'img_volus_{}'.format(self.acc_factor)
        #self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #for acc_factor in acc_factors:

        #files = list(pathlib.Path(os.path.join(root,'acc_{}'.format(acc_factor))).iterdir())

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type,dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 

            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            acc_val = float(acc_factor[:-1].replace("_","."))
            mask_val = 0 if mask_type=='cartesian' else 1
            dataset_val = 0 if dataset_type=='mrbrain_t1' else 1
            gamma_input = np.array([acc_val, mask_val,dataset_val])
            #gamma_input = np.array([acc_val,dataset_val])
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),str(fname.name),slice, torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type
            #return torch.from_numpy(zf_img), torch.from_numpy(target),str(fname.name),slice

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,dataset_type,mask_type,acc_factor):

        # List the h5 files in root 
        newroot = os.path.join(root, dataset_type,mask_type,'validation','acc_{}'.format(acc_factor))
        files = list(pathlib.Path(newroot).iterdir())
        self.examples = []
        #self.acc_factor = acc_factor
        #self.dataset_type = dataset_type

        #self.key_img = 'img_volus_{}'.format(self.acc_factor)
        #self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #for acc_factor in acc_factors:

        #files = list(pathlib.Path(os.path.join(root,'acc_{}'.format(acc_factor))).iterdir())
        #print(files)
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type,dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            #target = data['volfs'][:,:,slice]
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 

            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            acc_val = float(acc_factor[:-1].replace("_","."))
            mask_val = 0 if mask_type=='cartesian' else 1
            dataset_val = 0 if dataset_type=='mrbrain_t1' else 1
            gamma_input = np.array([acc_val, mask_val,dataset_val])
            #gamma_input = np.array([acc_val,dataset_val])
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type
 
