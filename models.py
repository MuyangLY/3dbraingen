import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sp_norm
import copy
from utils import to_decision, get_norm_by_name


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_channels(which_net, base_multipler):
    channel_multipliers = {
        "Generator": [8, 8, 8, 8, 8, 8, 8, 4, 2, 1],
        "Discriminator": [1, 2, 4, 8, 8, 8, 8, 8, 8]
    }
    ans = list()
    for item in channel_multipliers[which_net]:
        ans.append(int(item * base_multipler))
    return ans


class Discriminator_test(nn.Module):
    def __init__(self):
        super(Discriminator_test, self).__init__()
        self.num_blocks = 6
        self.num_blocks_ll = 2
        self.norm_name = "batch"
        # self.prob_FA = {"content": config_D["prob_FA_con"], "layout": config_D["prob_FA_lay"]}
        # self.no_masks = config_D["no_masks"]
        self.num_mask_channels = 5
        self.bernoulli_warmup = 10000
        num_of_channels = get_channels("Discriminator", 32)[:self.num_blocks + 1]  # [:8] [32, 64, 128, 256, 256, 256, 256, 256, 256, 256]
        
        for i in range(self.num_blocks_ll+1, self.num_blocks):  # [3:6]
            num_of_channels[i] = int(num_of_channels[i] * 2)  # [32, 64, 128, 256, 256, 512, 512, 256]
        self.feature_prev_ratio = 8  # for msg concatenation  多尺度梯度连结  允许梯度流从鉴别器到发生器多个尺度上流动

        self.body_ll, self.body_content, self.body_layout = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        self.rgb_to_features = nn.ModuleList([])  # for msg concatenation
        self.final_ll, self.final_content, self.final_layout = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])

        # --- D low-level --- #
        for i in range(self.num_blocks_ll):  # 构造
            if i>0:
                cur_block = D_block(num_of_channels[i-1], num_of_channels[i], self.norm_name, is_first=i == 0)
            else:
                cur_block = D_block(1, num_of_channels[i], self.norm_name, is_first=i == 0)
            self.body_ll.append(cur_block)

            self.final_ll.append(to_decision(num_of_channels[i], 1))

        # --- D content --- #
        # self.content_FA = Content_FA(self.no_masks, self.prob_FA["content"], self.num_mask_channels)  # 变换图像
        for i in range(self.num_blocks_ll, self.num_blocks):  # [2，6]
            k = i - self.num_blocks_ll
            cur_block_content = D_block(num_of_channels[i - 1], num_of_channels[i], self.norm_name, only_content=True)  # （64， 128）
            self.body_content.append(cur_block_content)
            out_channels = self.num_mask_channels + 1
            self.final_content.append(to_decision(num_of_channels[i], out_channels))  # k = s = 1

        # --- D layout --- #
        # self.layout_FA = Layout_FA(self.no_masks, self.prob_FA["layout"])
        for i in range(self.num_blocks_ll, self.num_blocks):  # [4，7]
            k = i - self.num_blocks_ll  # 1
            in_channels = 1 if k > 0 else num_of_channels[i - 1]
            cur_block_layout = D_block(in_channels, 1, self.norm_name)  # （1， 1）
            self.body_layout.append(cur_block_layout)
            self.final_layout.append(to_decision(1, 1))
        print("Created Discriminator (%d+%d blocks) with %d parameters" %
              (self.num_blocks_ll, self.num_blocks-self.num_blocks_ll, sum(p.numel() for p in self.parameters())))

    def content_masked_attention(self, y, mask, for_real, epoch):
        mask = F.interpolate(mask, size=(y.shape[2], y.shape[3], y.shape[4]), mode="nearest")  # y为Dll的输出
        y_ans = torch.zeros_like(y).repeat(mask.shape[1], 1, 1, 1, 1)  # .repeat()复制x次维度，1代表不复制
        if not for_real:  # 生成的掩膜遵循伯努利分布
            mask_soft = mask
            if epoch < self.bernoulli_warmup:  # default 10000
                mask_hard = torch.bernoulli(torch.clamp(mask, 0.001, 0.999))  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
            else:
                mask_hard = F.one_hot(torch.argmax(mask, dim=1)).permute(0, 4, 1, 2, 3)  # 返回最大的索引
            mask = mask_hard - mask_soft.detach() + mask_soft  # https://blog.csdn.net/orangerfun/article/details/116211051 .data和.detach() 分离数据，但是推荐使用detach()，因为其更安全，
        for i_ch in range(mask.shape[1]):
            y_ans[i_ch * (y.shape[0]):(i_ch + 1) * (y.shape[0])] = mask[:, i_ch:i_ch + 1, :, :, :] * y
        return y_ans

    def forward(self, images, masks,for_real, epoch):  # input 是生成器生成的
        # images = inputs["images"]
        # masks = inputs["masks"] if not self.no_masks else None
        output_ll, output_content, output_layout = list(), list(), list(),

        # --- D low-level --- #
        # y = self.rgb_to_features[0](images[-1])
        y = images
        for i in range(0, self.num_blocks_ll):  # [0,2]
            # if i > 0:
            #     y = torch.cat((y, self.rgb_to_features[i](images[-i - 1])), dim=1)   # 连接两个层
            y = self.body_ll[i](y)
            output_ll.append(self.final_ll[i](y))

        # --- D content --- #
        y_con = y
        
        y_con = self.content_masked_attention(y, masks, for_real, epoch)  # y --> D low-level
        y_con = torch.mean(y_con, dim=(2, 3, 4), keepdim=True)
        # if for_real:
        #     y_con = self.content_FA(y_con)
        for i in range(self.num_blocks_ll, self.num_blocks):  # [4,7]
            k = i - self.num_blocks_ll
            y_con = self.body_content[k](y_con)
            output_content.append(self.final_content[k](y_con))

        # --- D layout --- #
        y_lay = y
        # if for_real:
        #     y_lay = self.layout_FA(y, masks)
        for i in range(self.num_blocks_ll, self.num_blocks):
            k = i - self.num_blocks_ll
            y_lay = self.body_layout[k](y_lay)
            output_layout.append(self.final_layout[k](y_lay))

        return {"low-level": output_ll, "content": output_content, "layout": output_layout}


class D_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first=False, only_content=False):
        super(D_block, self).__init__()
        middle_channel = min(in_channel, out_channel)
        ker_size, padd_size = (1, 0) if only_content else (3, 1)
        self.is_first = is_first
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = sp_norm(nn.Conv3d(in_channel, middle_channel, ker_size, padding=padd_size))
        self.conv2 = sp_norm(nn.Conv3d(middle_channel, out_channel, ker_size, padding=padd_size))
        self.norm1 = get_norm_by_name(norm_name, in_channel)
        self.norm2 = get_norm_by_name(norm_name, middle_channel)
        self.down = nn.AvgPool3d(2) if not only_content else torch.nn.Identity()
        learned_sc = in_channel != out_channel or not only_content
        if learned_sc:
            self.conv_sc = sp_norm(nn.Conv3d(in_channel, out_channel, (1, 1, 1), bias=False))
        else:
            self.conv_sc = torch.nn.Identity()

    def forward(self, x):
        h = x
        if not self.is_first:
            x = self.norm1(x)
            x = self.activ(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        if not x.shape[0] == 0:
            x = self.down(x)
        h = self.conv_sc(h)
        if not x.shape[0] == 0:
            h = self.down(h)
        return x + h
    


class Discriminator_w2(nn.Module):
    def __init__(self):
        super(Discriminator_test, self).__init__()
        self.num_blocks = 6
        self.norm_name = "batch"
        # self.prob_FA = {"content": config_D["prob_FA_con"], "layout": config_D["prob_FA_lay"]}
        # self.no_masks = config_D["no_masks"]
        self.num_mask_channels = 5
        self.bernoulli_warmup = 10000
        num_of_channels = get_channels("Discriminator", 32)[:self.num_blocks + 1]  # [:8] [32, 64, 128, 256, 256, 256, 256, 256, 256, 256]
        
        for i in range(self.num_blocks):  # [3:6]
            num_of_channels[i] = int(num_of_channels[i] * 2)  # [32, 64, 128, 256, 256, 512, 512, 256]
        # self.feature_prev_ratio = 8  # for msg concatenation  多尺度梯度连结  允许梯度流从鉴别器到发生器多个尺度上流动

        self.body_content = nn.ModuleList([])
        # self.rgb_to_features = nn.ModuleList([])  # for msg concatenation
        self.final_content = nn.ModuleList([])

        # # --- D low-level --- #
        # for i in range(self.num_blocks_ll):  # 构造
        #     if i>0:
        #         cur_block = D_block(num_of_channels[i-1], num_of_channels[i], self.norm_name, is_first=i == 0)
        #     else:
        #         cur_block = D_block(1, num_of_channels[i], self.norm_name, is_first=i == 0)
        #     self.body_ll.append(cur_block)

        #     self.final_ll.append(to_decision(num_of_channels[i], 1))

        # --- D content --- #
        # self.content_FA = Content_FA(self.no_masks, self.prob_FA["content"], self.num_mask_channels)  # 变换图像
        for i in range(self.num_blocks):  # [2，6]

            cur_block_content = D_block(num_of_channels[i], num_of_channels[i+1], self.norm_name, only_content=True)  # （64， 128）
            self.body_content.append(cur_block_content)
            out_channels = self.num_mask_channels + 1
            self.final_content.append(to_decision(num_of_channels[i+1], out_channels))  # k = s = 1

        # --- D layout --- #
        # self.layout_FA = Layout_FA(self.no_masks, self.prob_FA["layout"])
        # for i in range(self.num_blocks_ll, self.num_blocks):  # [4，7]
        #     k = i - self.num_blocks_ll  # 1
        #     in_channels = 1 if k > 0 else num_of_channels[i - 1]
        #     cur_block_layout = D_block(in_channels, 1, self.norm_name)  # （1， 1）
        #     self.body_layout.append(cur_block_layout)
        #     self.final_layout.append(to_decision(1, 1))
        # print("Created Discriminator (%d+%d blocks) with %d parameters" %
        #     (self.num_blocks_ll, self.num_blocks-self.num_blocks_ll, sum(p.numel() for p in self.parameters())))

    def content_masked_attention(self, y, mask, for_real, epoch):
        mask = F.interpolate(mask, size=(y.shape[2], y.shape[3], y.shape[4]), mode="nearest")  # y为Dll的输出
        y_ans = torch.zeros_like(y).repeat(mask.shape[1], 1, 1, 1, 1)  # .repeat()复制x次维度，1代表不复制
        if not for_real:  # 生成的掩膜遵循伯努利分布
            mask_soft = mask
            if epoch < self.bernoulli_warmup:  # default 10000
                mask_hard = torch.bernoulli(torch.clamp(mask, 0.001, 0.999))  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
            else:
                mask_hard = F.one_hot(torch.argmax(mask, dim=1)).permute(0, 4, 1, 2, 3)  # 返回最大的索引
            mask = mask_hard - mask_soft.detach() + mask_soft  # https://blog.csdn.net/orangerfun/article/details/116211051 .data和.detach() 分离数据，但是推荐使用detach()，因为其更安全，
        for i_ch in range(mask.shape[1]):
            y_ans[i_ch * (y.shape[0]):(i_ch + 1) * (y.shape[0])] = mask[:, i_ch:i_ch + 1, :, :, :] * y
        return y_ans

    def forward(self, images, masks, for_real, epoch):  # input 是生成器生成的
        # images = inputs["images"]
        # masks = inputs["masks"] if not self.no_masks else None
        output_content = list()

        # --- D low-level --- #
        # y = self.rgb_to_features[0](images[-1])
        y = images
        # for i in range(0, self.num_blocks_ll):  # [0,2]
        #     # if i > 0:
        #     #     y = torch.cat((y, self.rgb_to_features[i](images[-i - 1])), dim=1)   # 连接两个层
        #     y = self.body_ll[i](y)
        #     output_ll.append(self.final_ll[i](y))

        # --- D content --- #

        y_con = self.content_masked_attention(y, masks, for_real, epoch)  # y --> D low-level
        y_con = torch.mean(y_con, dim=(2, 3, 4), keepdim=True)
        # if for_real:
        #     y_con = self.content_FA(y_con)
        for i in range(self.num_blocks):  # [2,6]

            y_con = self.body_content[i](y_con)
            output_content.append(self.final_content[i](y_con))

        # # --- D layout --- #
        # y_lay = y
        # # if for_real:
        # #     y_lay = self.layout_FA(y, masks)
        # for i in range(self.num_blocks_ll, self.num_blocks):
        #     k = i - self.num_blocks_ll
        #     y_lay = self.body_layout[k](y_lay)
        #     output_layout.append(self.final_layout[k](y_lay))

        return output_content
        # return {"low-level": output_ll, "content": output_content, "layout": output_layout}
