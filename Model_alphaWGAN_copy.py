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

class Discriminator_content(nn.Module):
    def __init__(self, channel=512,out_class=4,is_dis =True):
        super(Discriminator_content, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class
        self.bernoulli_warmup = 15000
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        

    def content_masked_attention(self, y, mask, epoch):
        mask = F.interpolate(mask, size=(y.shape[2], y.shape[3], y.shape[4]), mode="nearest")  # 上/下采样
        y_ans = torch.zeros_like(y).repeat(mask.shape[1], mask.shape[1], 1, 1, 1)  # .repeat()复制x次维度，1代表不复制 mask.shape[1]=4
        # 生成的掩膜遵循伯努利分布
        mask_soft = mask
        if epoch < self.bernoulli_warmup:  # default 15000
            mask_hard = torch.bernoulli(torch.clamp(mask, 0.001, 0.999))  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
        else:
            mask_hard = F.one_hot(torch.argmax(mask, dim=1), num_classes=mask_soft.shape[1]).permute(0, 4, 1, 2, 3)  # 返回最大的索引
        mask = mask_hard - mask_soft.detach() + mask_soft  # https://blog.csdn.net/orangerfun/article/details/116211051 .data和.detach() 分离数据，但是推荐使用detach()，因为其更安全，
        for i_ch in range(mask.shape[1]):
            y_ans[i_ch * (y.shape[0]):(i_ch + 1) * (y.shape[0])] = mask[:, i_ch:i_ch + 1, :, :, :] * y
        return y_ans

    def forward(self, x, mask, epoch, _return_activations=False):
        x = self.content_masked_attention(x, mask, epoch)
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
        # n_class = 1
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
        
        self.mask_coverter = nn.Conv3d(1, 5, kernel_size=3, padding=1, bias=True)  # 生成掩膜 (4, 5, 64, 64, 64)

    def forward(self, noise):

        # output = dict()

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

        image = F.tanh(h)
        # output["image"] = image

        mask = self.mask_coverter(h)
        mask = F.softmax(mask, dim = 2)
        # output["mask"] = mask

        return image, mask
