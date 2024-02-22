import torch
import torch.nn as nn
import torch.nn.functional as F

class NonNegativeConv2d(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(NonNegativeConv2d, self).__init__()
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size,kernel_size))
        nn.init.kaiming_uniform_(self.weight) #xavier_uniform(self.weight)
        #print(self.weight)
        if bias==False:
            self.bias=None
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
    
          
    
    def forward(self, x):
        #weight = torch.relu(self.weight)  # Applying ReLU to ensure non-negativity
        weight=self.weight.clone()
        weight.clamp_(0, float('inf'))
        return nn.functional.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)


class bsconv_layer(nn.Module):

    def __init__(self, inplanes, outplanes, scale_factor=None):
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        #self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.conv1=NonNegativeConv2d(2*inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        
        #x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv1(torch.cat((self.relu(self.bn1(x)), -self.relu(self.bn1(x))), dim=1))

        return x