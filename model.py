import torch.nn as nn
import torch
import numpy as np
#learns features from image and gives potential action outputs
class Network(nn.Module):
    def __init__(self, frames, num_actions, batch_size=1):
        super(Network, self).__init__()
        self.out_channels = [32, 64, 64]
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]
        #learn features from image with conv2d
        self.conv_stack = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        h_out, w_out = self._last_conv_out_size(frames[1], frames[2], num_convs=3)
        #take features and map them to outputs (actions)
        self.fc = nn.Sequential(
            nn.Linear(h_out*w_out*batch_size*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    #returns the output sizes of a conv layer
    def _conv_out_size(self, h_in, w_in, kernel_size=1, stride=1, padding=0):
        out = lambda o: (o - (2*padding) - kernel_size)//stride + 1
        return out(h_in), out(w_in)
    
    #returns the final convolutional layer's output sizes
    def _last_conv_out_size(self, h_in, w_in, num_convs=1):#feed in image height, width and the number of conv layers
        for conv in range(num_convs):
            h_in, w_in = self._conv_out_size(h_in, w_in, self.kernel_sizes[conv], self.strides[conv])
        h_out_size, w_out_size = h_in, w_in
        return h_out_size, w_out_size
    
    def forward(self, imgs_tensor):
        #print(imgs_tensor.shape)
        imgs_tensor = self.conv_stack(imgs_tensor)
        imgs_tensor = imgs_tensor.view(1, -1)# Flatten to 1xm
        imgs_tensor = self.fc(imgs_tensor)
        return imgs_tensor

