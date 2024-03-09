import torch.nn as nn
from torch.nn import functional as F  # Import functional for activation functions
import copy

class cnn(nn.Module):
    def __init__(self, frames, num_actions):
        super(cnn, self).__init__()
        self.out_channels = [32, 64, 64]
        self.kernel_sizes = [8, 4, 3]
        self.strides = [4, 2, 1]
        self.conv_stack = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        h_out, w_out = self._last_conv_out_size(frames[1], frames[2], num_convs=3)
        self.fc = nn.Sequential(
            nn.Linear(h_out*w_out*64, 512),
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
    
    def forward(self, img_tensor):
        img_tensor = self.conv_stack(img_tensor)
        img_tensor = img_tensor.view(1, -1)# Flatten to 1xm
        img_tensor = self.fc(img_tensor)
        return img_tensor

