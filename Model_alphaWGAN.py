import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

#***********************************************
#Encoder and Discriminator has same architecture
#***********************************************
class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)  # c1->LeakReLU    negative_slope：x为负数时的需要的一个系数，控制负斜率的角度。默认值：1e-2
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)  # c2->bn2->LeakReLU 
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)  # c3->bn3->LeakReLU
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)  # c4->bn4->LeakReLU
        h5 = self.conv5(h4)
        output = h5
        
        return output
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))  # 相当于x < 0 x<0x<0，LeakyReLU(x) = x （与大于0时一致了，输入是啥，输出就是啥）
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
            
        return output

class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*8, kernel_size=4, stride=1, padding=0, bias=False)  # 向输出添加可学习的偏差。默认值：True
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        h = self.relu(self.bn1(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.2):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('in', x.shape)
        # print('out', self.model(x).shape)
        return self.model(x)

class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        x = nn.functional.pad(x, (1,0,1,0,1,0))

        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('new')
        # print(x.shape)
        # print(skip_input.shape)
        x = self.model(x)
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.mid1 = UNetMid(1024, 512, dropout=0.2)
        self.mid2 = UNetMid(1024, 512, dropout=0.2)
        self.mid3 = UNetMid(1024, 512, dropout=0.2)
        self.mid4 = UNetMid(1024, 256, dropout=0.2)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        # self.us =   nn.Upsample(scale_factor=2)

        self.final = nn.ConvTranspose3d(128, out_channels, 4, 2, 1)
            # nn.Conv3d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m1 = self.mid1(d4, d4)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        # u7 = self.up7(u6, d1)
        # u7 = self.us(u7)
        # u7 = nn.functional.pad(u7, pad=(1,0,1,0,1,0))
        # # print(self.final(u7).shape)
        o1 = self.final(u3)
        o2 = F.tanh(o1)
        return o2
