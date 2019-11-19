DEBUG=False
def log(s):
    if DEBUG:
        print(s)
###################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def regression_l1(input, target, weight=None, size_average=True):
    # loss = nn.L1Loss(input, target, size_average=size_average)
    loss = nn.L1Loss(input, target)
    return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    # print('input: ', input.size())
    # print('target: ', target.size())
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1).long()
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )

    # print(type(loss))
    return loss


# def dice_loss(input, target, weight=None, size_average=False):
#     smooth = 1.
#
#     # print(input.size())
#     # print(target.size())
#
#     iflat = input.data.max(1)[1].view(-1)
#     print(iflat.size())
#     tflat = target.view(-1)
#     print(tflat.size())
#     intersection = (iflat * tflat).sum()
#     dice = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))
#     # dice.requires_grad = True
#
#     return 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))


# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]
#
# class DiceLoss(Function):
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def forward(self, input, target, save=True):
#         print(input.shape, target.shape)
#
#         eps = 0.000001
#         # _, result_ = input.max(1)
#
#         print(input.data.shape)
#         result = input.data.max(1)[1].float()
#         print(result.size())
#         if save:
#             self.save_for_backward(result, target)
#         print('result:', result.size())
#         self.target_ = torch.cuda.FloatTensor(target.size())
#         # result.copy_(result_)
#         target = self.target_.view(-1)
#         result = result.view(-1)
#         print('input:', input.size(), 'target:', target.size())
#         print('result:', result.size())
#         # result_ = torch.squeeze(result_)
#         # if input.is_cuda:
#         #     result = torch.cuda.FloatTensor(result_.size())
#         #     self.target_ = torch.cuda.FloatTensor(target.size())
#         # else:
#         #     result = torch.FloatTensor(result_.size())
#         #     self.target_ = torch.FloatTensor(target.size())
#         # result = result.copy_(result_.view(-1))
#         # self.target_ = target.copy()
#         # target = self.target_
# #       print(input)
#         print(result.size(), target.size())
#
#         for v in self.saved_tensors:
#             print('saved:', v.size())
#         # print(self.saved_tensors.size())
#
#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#             union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         self.intersect, self.union = intersect, union
#         return out
#
#     def backward(self, grad_output):
#         input, _ = self.saved_tensors
#         intersect, union = self.intersect, self.union
#         target = self.target_
#         print('inter', intersect.size(), 'un:', union.size())
#         print('target', target.size())
#         gt = torch.div(target, union)
#         print('gt', gt.size())
#         IoU2 = intersect/(union*union)
#         print(IoU2)
#         print(input.size())
#         pred = torch.mul(input, IoU2)
#         print('pred', pred.size())
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input , None

# class DiceLoss(Function):
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def forward(self, input, target, save=True):
#         if save:
#             self.save_for_backward(input, target)
#         eps = 0.000001
#         _, result_ = input.max(1)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             self.target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             self.target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         self.target_.copy_(target)
#         target = self.target_
# #       print(input)
#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#             union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         self.intersect, self.union = intersect, union
#         return out
#
#     def backward(self, grad_output):
#         input, _ = self.saved_tensors
#         intersect, union = self.intersect, self.union
#         target = self.target_
#         gt = torch.div(target, union)
#         IoU2 = intersect/(union*union)
#         pred = torch.mul(input[:, 1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input , None
#
#
# def dice_loss(input, target):
#     return DiceLoss()(input, target)


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1.


        #iflat = pred.max(1)[1].view(-1).float()
        # print(pred.size(), target.size())
        iflat = F.softmax(pred, dim=1)[:,1,:].contiguous().view(-1)
        tflat = target.view(-1).float()
        intersection = (iflat * tflat).sum().float()
        dice_score = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return dice_score

def cross_entropy3d(input, target, weight=None, size_average=True):
    log('LOSS=>CrossEntropy3D=>input.size():{} target.size():{}'.format(input.size(), target.size()))
    loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
    return loss(input, target)

def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
