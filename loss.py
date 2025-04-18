import math
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np


def iou_aware_loss(iou_pred, actual_iou, mask):
    """
    iou_pred: [N] predicted IoUs in [0,1]
    actual_iou: [N] actual IoU for each box
    mask: [N] valid box mask
    """
    mask = mask > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=iou_pred.device, requires_grad=True)

    pred_iou = iou_pred[mask]
    true_iou = actual_iou[mask]
    # e.g. MSE loss
    return torch.nn.MSELoss()(pred_iou, true_iou)

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss



def ciou_loss(pred,target,weight,avg_factor=None,eps=1e-5):
        """CIoU loss.
        Computing the CIoU loss between a set of predicted bboxes and target bboxes.
        """
        """CIoU loss.
            Computing the CIoU loss between a set of predicted bboxes and target bboxes.
            """
        #pred = torch.clamp(pred, 1e-6, 128)


        pos_mask = weight > 0
        weight = weight[pos_mask].float()
        if avg_factor is None:
            avg_factor = torch.sum(pos_mask) + eps
        bboxes1 = torch.reshape(pred[pos_mask], (-1, 4)).float()
        bboxes2 = torch.reshape(target[pos_mask], (-1, 4)).float()



        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        wh = (rb - lt + 1).clamp(min=0.)


        overlap = wh[:, 0] * wh[:, 1]
        ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (ap + ag - overlap)

        # cal outer boxes
        outer_left_up = torch.min(bboxes1[:, :2], bboxes2[:, :2])
        outer_right_down = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        outer = (outer_right_down - outer_left_up).clamp(min=0)

        outer_diagonal_line = (outer[:, 0])**2 + (outer[:, 1])**2

        boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:] + 1) * 0.5
        boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:] + 1) * 0.5
        center_dis = (boxes1_center[:, 0] - boxes2_center[:, 0])**2 + \
                     (boxes1_center[:, 1] - boxes2_center[:, 1])**2

        boxes1_size = (bboxes1[:, 2:] - bboxes1[:, :2]).clamp(min=0)
        boxes2_size = (bboxes2[:, 2:] - bboxes2[:, :2]).clamp(min=0)

        v = (4.0 / (np.pi ** 2)) * \
            (torch.atan(boxes2_size[:, 0] / (boxes2_size[:, 1] + eps)) -
                      torch.atan(boxes1_size[:, 0] / (boxes1_size[:, 1] + eps)))**2

        S = (ious> 0.5).float()
        alpha = S * v / (1 - ious + v)

        cious = ious - (center_dis / outer_diagonal_line) - alpha * v

        cious = 1 - cious

        cious=cious*weight

        ## filter out the nan loss
        nan_index=~torch.isnan(cious)
        
        cious=cious[nan_index]


        return torch.sum(cious) / avg_factor

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']