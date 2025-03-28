import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import sys,os

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(pred_hms, pred_whs, pred_offsets,pred_ious, confidence=0.3, cuda=True):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        if pred_ious is not None:
            # shape could be [b, H, W] or [b, 1, H, W]
            iou_map = pred_ious[batch]
            if iou_map.dim() == 3 and iou_map.shape[0] == 1:
                iou_map = iou_map.squeeze(0)  
            # Now iou_map is [H, W]
            iou_map = iou_map.view(-1)  # flatten to [H*W]
        else:
            iou_map = None

        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv      = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        if iou_map is not None:
            iou_mask = iou_map[mask]   # shape [N_filtered]
            # Combine with class confidence
            final_conf = class_conf[mask] * iou_mask
        else:
            final_conf = class_conf[mask]

        detect = torch.cat([bboxes, torch.unsqueeze(final_conf,-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects

def centernet_correct_boxes_xyxy(decoded_boxes, input_shape, image_shape, letterbox_image):
    """
    Adjusts bounding boxes predicted by decode_bbox to fit the original image shape.
    
    Args:
        decoded_boxes (torch.Tensor): Tensor of shape (N, 4) with bounding boxes
                                       in the format [x_min, y_min, x_max, y_max].
        input_shape (tuple): Shape of the input image to the model (e.g., (512, 512)).
        image_shape (tuple): Shape of the original image (e.g., (height, width)).
        letterbox_image (bool): Whether letterboxing (padding) was applied during preprocessing.

    Returns:
        torch.Tensor: Adjusted bounding boxes in the same format as input.
    """
    input_shape = torch.tensor(input_shape, dtype=torch.float32)
    image_shape = torch.tensor(image_shape, dtype=torch.float32)

    decoded_boxes = decoded_boxes[0]
    
    if len(decoded_boxes) > 0:
        if letterbox_image:
            # Calculate scaling and offset for letterboxed image
            new_shape = torch.round(image_shape * torch.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            # Adjust bounding boxes
            decoded_boxes[:, 0] = (decoded_boxes[:, 0] / input_shape[1] - offset[1]) * scale[1]
            decoded_boxes[:, 2] = (decoded_boxes[:, 2] / input_shape[1] - offset[1]) * scale[1]
            decoded_boxes[:, 1] = (decoded_boxes[:, 1] / input_shape[0] - offset[0]) * scale[0]
            decoded_boxes[:, 3] = (decoded_boxes[:, 3] / input_shape[0] - offset[0]) * scale[0]
        else:
            # Adjust bounding boxes directly for scaled images
            decoded_boxes[:, 0] /= input_shape[1]
            decoded_boxes[:, 2] /= input_shape[1]
            decoded_boxes[:, 1] /= input_shape[0]
            decoded_boxes[:, 3] /= input_shape[0]

        # Scale back to original image shape
        decoded_boxes[:, 0] *= image_shape[1]
        decoded_boxes[:, 2] *= image_shape[1]
        decoded_boxes[:, 1] *= image_shape[0]
        decoded_boxes[:, 3] *= image_shape[0]

        # Clip boxes to image boundaries
        decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], 0, image_shape[1])
        decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], 0, image_shape[1])
        decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], 0, image_shape[0])
        decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], 0, image_shape[0])
    else:
        decoded_boxes = torch.tensor(decoded_boxes)

    return decoded_boxes

def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    
    #----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        if len(detections) == 0:
            continue
        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels   = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # #------------------------------------------#
                # #   按照存在物体的置信度排序
                # #------------------------------------------#
                # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # #------------------------------------------#
                # #   进行非极大抑制
                # #------------------------------------------#
                # max_detections = []
                # while detections_class.size(0):
                #     #---------------------------------------------------#
                #     #   取出这一类置信度最高的，一步一步往下判断。
                #     #   判断重合程度是否大于nms_thres，如果是则去除掉
                #     #---------------------------------------------------#
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # #------------------------------------------#
                # #   堆叠
                # #------------------------------------------#
                # max_detections = torch.cat(max_detections).data
            else:
                max_detections  = detections_class
            
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output