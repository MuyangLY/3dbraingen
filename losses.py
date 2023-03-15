import torch
import torch.nn.functional as F


class losses_computer():

    def content_segm_loss(self, out_d, data, real):
        """
        The multi-class cross-entropy loss used in the content masked attention
        """
        mask = data
        mask_ch = mask.shape[1]
        if real:
            ground_t = torch.arange(mask_ch).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [4, 1, 1, 1, 1]
            ground_t = ground_t.repeat(1, 1, out_d.shape[2], out_d.shape[3], out_d.shape[4])
            ground_t = ground_t.repeat_interleave(mask.shape[0], dim=0)[:, 0, :, :, :]  # 沿着指定的维度重复张量的元素
        else:  # fake
            ground_t = torch.ones_like(out_d)[:, 0, :, :, :] * mask_ch
        # weights = torch.cat((1 / (torch.sum(mask.detach(), dim=(0, 2, 3, 4))), torch.Tensor([1.0]).to(out_d.device)))
        weights = 1 / (torch.sum(mask.detach(), dim=(0, 1, 2, 3, 4)))
        weights[weights == float('inf')] = 0
        # weights = weights.max()
        loss = F.cross_entropy(out_d, ground_t.long().to(out_d.device), weight=weights.to(out_d.device))
        return loss

    def __call__(self, out_d, data, real):

        # --- adversarial loss ---#
        losses = self.content_segm_loss(out_d, data, real)  # .get() 返回键值并赋值0

        return losses


def wgan_loss(output, real, forD):
    if real and forD:
        ans = -output.mean()
    elif not real and forD:
        ans = output.mean()
    elif real and not forD:
        ans = -output.mean()
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    #print(real, forD, ans)
    return ans


def hinge_loss(output, real, forD):
    if real and forD:
        minval = torch.min(output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif not real and forD:
        minval = torch.min(-output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif real and not forD:
        ans = -torch.mean(output)
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    return ans


def bce_loss(output, real, forD, no_aggr=False):
    target_tensor = get_target_tensor(output, real).to(output.device)
    ans = F.binary_cross_entropy_with_logits(output, target_tensor, reduction=("mean" if not no_aggr else "none"))
    return ans


def get_target_tensor(input, target_is_real):
    if target_is_real:
        real_label_tensor = torch.FloatTensor(1).fill_(1)
        real_label_tensor.requires_grad_(False)
    else:
        real_label_tensor = torch.FloatTensor(1).fill_(0)
        real_label_tensor.requires_grad_(False)
    return real_label_tensor.expand_as(input)


def get_zero_tensor(input):
    zero_tensor = torch.FloatTensor(1).fill_(0)
    zero_tensor.requires_grad_(False)
    return zero_tensor.expand_as(input)
