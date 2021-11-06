import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

class DataConsistencyLayer(nn.Module):

    def __init__(self, us_mask_path, device):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask_path = us_mask_path
        self.device = device

    def forward(self,predicted_img,us_kspace,acc_factor, mask_string, dataset_string):
        
        us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))
        #print("us_mask_path: ", us_mask_path)
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        #print(us_mask.dtype)
        updated_kspace1  = us_mask * us_kspace 
        updated_kspace2  = (1 - us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1 + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

class MACReconNet(nn.Module):
    
    def __init__(self, args,n_ch=1):
        
        super(MACReconNet,self).__init__()
        

        self.relu = nn.ReLU() 
        self.weights = {'fc1':[32,1,3,3],
                        'fc2':[32,32,3,3],
                        'fc3':[32,32,3,3],
                        'fc4':[32,32,3,3],
                        'fc5':[1,32,3,3]}
        

        cascade  = nn.ModuleDict({
            'fc1':nn.Linear(3,np.prod(self.weights['fc1'])),
            'fc2':nn.Linear(3,np.prod(self.weights['fc2'])),
            'fc3':nn.Linear(3,np.prod(self.weights['fc3'])),
            'fc4':nn.Linear(3,np.prod(self.weights['fc4'])),
            'fc5':nn.Linear(3,np.prod(self.weights['fc5']))})

        instance_norm = nn.ModuleDict({
            'fc1':nn.InstanceNorm2d(32,affine=True),
            'fc2':nn.InstanceNorm2d(32,affine=True),
            'fc3':nn.InstanceNorm2d(32,affine=True),
            'fc4':nn.InstanceNorm2d(32,affine=True) })
        dc = DataConsistencyLayer(args.usmask_path, args.device)
            
        self.layer = nn.ModuleList([cascade])
        #print(self.layer[0].keys())
        self.instance_norm = nn.ModuleList([instance_norm])
        
        self.dc = nn.ModuleList([dc])
         
        
    def forward(self,x, k, gamma_val, acc_string, mask_string, dataset_string):
    #def forward(self,x, acc):
        #print("x enter: ", x.size())
        batch_size = x.size(0)
        batch_outputs = []
        for n in range(batch_size):        
            xout = x[n]
            xout = xout.unsqueeze(0)
            #print("xout shape in: ",xout.shape)
            xtemp = xout

            for fc_no in self.layer[0].keys():
                #print(fc_no)    
                conv_weight = self.layer[0][fc_no](gamma_val[n])
                conv_weight = torch.reshape(conv_weight,self.weights[fc_no])

                if fc_no=='fc5':
                    xout = F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)
                else:
                    xout = self.relu(self.instance_norm[0][fc_no](F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)))
                    #print("xout shape: ",xout.shape)
            
            xout = xout + xtemp
            xout = self.dc[0](xout,k[n],acc_string[n], mask_string[n], dataset_string[n])
             
            #print("xout shape out: ",xout.shape)
            batch_outputs.append(xout)
        output = torch.cat(batch_outputs, dim=0)
        return output  


class DC_CNN(nn.Module):
    
    def __init__(self, args, checkpoint_file, n_ch=1,nc=5):
    #def __init__(self, args, n_ch=1,nc=5):
        
        super(DC_CNN,self).__init__()
        
        cnn_blocks = []
        #dc_blocks = []
        checkpoint = torch.load(checkpoint_file)
        self.nc = nc
        
        for ii in range(self.nc): 
            
            cnn = MACReconNet(args)
            #cnn.load_state_dict(checkpoint['model'])  # uncomment this line to load the best model of MACReconNet
            cnn_blocks.append(cnn)
            
        
        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        
    def forward(self,x, k, gamma_val, acc_string, mask_string, dataset_string):
        x_cnn = x
        for i in range(self.nc):
            x_cnn = self.cnn_blocks[i](x_cnn, k, gamma_val, acc_string, mask_string, dataset_string)
        return x_cnn  


