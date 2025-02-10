"""
Created on Mon Feb 27 14:50:55 2023

@authors: mathiasrammhaugland, nc619

Setup from: https://github.com/aliasvishnu/EEGNet
Copy of updated TF model from EEGNet creators: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

Inspired by this PyTorch implementation (by Schirrmeister??):
https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py
"""

##### ----------------------------------------------------------------------------------------------------------------  #####
#                                                                                                                           #
# THIS PYTHON SCRIPT CONTAINS THE EEGNET MODEL IMPLEMENTATION IN PYTORCH. IT HAS BEEN ALTERED TO WORK WITH RAY TUNING.      #
#                                                                                                                           #
##### ----------------------------------------------------------------------------------------------------------------  #####


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d with weight constraint such that their norm doesn't exceed max_norm, for numerical stability
    """
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)




class EEGNet(nn.Module):
    def __init__(self, config = dict(), tune_bool =False, num_classes=2,
                 num_channels=6, seg_len=300, device="cpu",
                 net_params = {'fc_dropout': 0.5,
                               'conv_dropout': 0.5,
                               'L1': 0,
                               'L2': 0,
                               'lr': 0.0008,
                               'optim': 'Adam',
                               'F1': 8,
                               'D': 2}
                 , tune_verbose = True):
        """ EEGNet model

        Args:
            config (dict, optional): Ray config dictionary that has the hyperparameters being tuned. Defaults to dict().
            tune_bool (bool, optional): Whether to tune. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 2.
            num_channels (int, optional): Number of channels per segment. Defaults to 6.
            seg_len (int, optional): Length of segment in samples. Defaults to 300.
            device (str, optional): 'cpu', 'cuda', or 'mps'. Defaults to "cpu".
            net_params (dict, optional): Hyperparameters of the network. Defaults to {'fc_dropout': 0.5,'conv_dropout': 0.5,'L1': 0,'L2': 0,'lr': 0.0008,'optim': 'Adam','F1': 8,'D': 2}.
        """
        super(EEGNet, self).__init__()
        self.divide_fac = 1+seg_len//32 #Divide factor in time, e.g. 900=29, 600=19, 300=10, 120 = 4
        # Initialise model dimensions
        if not tune_bool:
            self.F1 = net_params['F1']
            self.D = net_params['D']
            self.F2 = self.F1 * self.D
        else:
            if 'F1' in config:
                if tune_verbose: print("F1 being configured...")
                self.F1 = int(config['F1'])
            else:
                self.F1 = net_params['F1']
            if 'D' in config:
                self.D = int(config['D'])
                if tune_verbose: print("D being configured...")
            else:
                self.D = net_params['D']
            if 'F2' in config:
                if tune_verbose: print("F2 being configured...")
                self.F2 = int(config['F2'])
            else:
                self.F2 = self.F1 * self.D
                
        self.device = device
        # Time conv + batchnorm
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 150), padding = 'same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1, momentum=0.01, eps=0.001)
        
        # Channel conv + batchnorm + avg pool + dropout
        self.depthwise = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.D*self.F1, kernel_size=(num_channels,1), stride=(num_channels,1), groups=self.F1, bias=False, max_norm=1)
        self.batchnorm2 = nn.BatchNorm2d(self.D*self.F1, momentum=0.01, eps=0.001)
        
        self.pooling1 = nn.AvgPool2d((1, 4))  #(1,28), (1,19), (1,10)
        if not tune_bool:
            self.dropout1 = nn.Dropout(p=net_params['conv_dropout'])
            self.dropout2 = nn.Dropout(p=net_params['conv_dropout'])
        else:
            if 'conv_dropout' in config:
                if tune_verbose: print("conv_dropout being configured...")
                self.dropout1 = nn.Dropout(p=config['conv_dropout'])
                self.dropout2 = nn.Dropout(p=config['conv_dropout'])
            else:
                if tune_verbose: print("conv_dropout not being configured...")
                self.dropout1 = nn.Dropout(p=net_params['conv_dropout'])
                self.dropout2 = nn.Dropout(p=net_params['conv_dropout'])
        
        # Time conv + pointwise conv + batchnorm + avg pool + dropout
        self.conv3 = nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.F2, kernel_size=(1, 40), groups=self.F2, padding='same',bias=False)
        self.pointwise = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(1,1), padding=(0,0),bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2, momentum=0.01, eps=0.001)
        self.pooling3 = nn.AvgPool2d((1, 8), ceil_mode=True)
                
        # FC Layer 1: conv -> 2D representation
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc0 = nn.Linear(self.F2*self.divide_fac, 10) #10 chosen arbitrarily
        
        # FC Layer 2: distance -> classification
        self.fc1 = nn.Linear(10+2,num_classes)
        if not tune_bool:
            self.dropout3 = nn.Dropout(p=net_params['fc_dropout'])
        else:
            if 'fc_dropout' in config:
                if tune_verbose: print("fc_dropout being configured...")
                self.dropout3 = nn.Dropout(p=config['fc_dropout'])
            else:
                if tune_verbose: print("fc_dropout not being configured...")
                self.dropout3 = nn.Dropout(p=net_params['fc_dropout'])

    def forward(self, x):
        # Layer 1
        x = self.conv1(x) 
        x = self.batchnorm1(x)

        x = self.depthwise(x)
        x = self.batchnorm2(x) 
        x = F.elu(x) #ELU activation
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.conv3(x)
        x = self.pointwise(x)
        x = self.batchnorm3(x)
        
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout2(x)
        
        # FC Layer
        x1 = torch.flatten(x[:,:,0,:], start_dim=1)
        x2 = torch.flatten(x[:,:,1,:], start_dim=1)

        #Reduce each input feature map to 10 dimensions using a dense layer
        x1 = self.fc0(x1)
        x1 = self.dropout3(x1)
        x2 = self.fc0(x2)
        x2 = self.dropout3(x2)
        # print(f"x1: {x1.shape}, x2: {x2.shape}")
        #Elementwise distance between the 10D tensors
        x = torch.abs(torch.sub(x1,x2,alpha=1)) # elementwise subtraction
        
        xdot = torch.diag(torch.tensordot(x1,x2,dims=([1],[1]))) #Dot 
        xdot = torch.unsqueeze(xdot,dim=1)

        xdist = F.pairwise_distance(x1,x2) # Distance between the two 10D tensors
        xdist = torch.unsqueeze(xdist,1)
        
        x = torch.cat((x,xdot,xdist),1)
        
        #Classifying layer
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        
        return x
