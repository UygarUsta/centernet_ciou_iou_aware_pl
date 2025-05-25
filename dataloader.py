import math

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils import extract_coordinates
from gaussan_functions import gaussian2D,gaussian_radius,draw_gaussian
import os 

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def resize_numpy(image, size, letterbox_image):
    image = np.array(image,dtype='float32')
    iw, ih = image.shape[1], image.shape[0]
    w, h = size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top+nh, left:left+nw, :] = resized_image
    else:
        new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return new_image

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std

def preprocess_input_simple(image):
    image   = image[:, :, ::-1] #np.array(image,dtype = np.float32)[:, :, ::-1]
    #mean    = [0.40789655, 0.44719303, 0.47026116]
    #std     = [0.2886383, 0.27408165, 0.27809834]
    return image / 255. #(image / 255. - mean) / std

class CenternetDataset(Dataset):
    def __init__(self, image_path, input_shape, classes, num_classes, train, stride=4,mosaic=True, mixup=True,center_sampling=True, center_sampling_radius=1):
        super(CenternetDataset, self).__init__()
        self.image_path = image_path
        self.length = len(self.image_path)
        self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.classes = classes
        self.num_classes = num_classes
        self.train = train
        self.mosaic = mosaic 
        self.mixup = mixup 
        # Center sampling için yeni parametreler
        self.center_sampling = center_sampling  # Center sampling kullanılsın mı?
        self.center_sampling_radius = center_sampling_radius  # Merkez etrafında kaç piksel işaretlenecek
        
    

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        self.mixup_prob = 0.25
        self.mixup_alpha = np.random.uniform(0.3, 0.5)
        self.mosaic_prob = 0.25
        
        use_mosaic = self.mosaic == True and np.random.rand() < self.mosaic_prob and self.train == True
        use_mixup = self.mixup == True  and  np.random.rand() <  self.mixup_prob and self.train == True
        

        # Load initial image and boxes
        if use_mosaic:
            image, box = self.get_mosaic_data(index)
        else:
            image, box = self.get_random_data(self.image_path[index], self.input_shape, random=self.train)

        # Apply Mixup augmentation
        if use_mixup:
            # Randomly select another image
            index2 = np.random.randint(0, self.length)
            # Load second image and boxes (with possible mosaic)
            if use_mosaic:
                image2, box2 = self.get_mosaic_data(index2)
            else:
                image2, box2 = self.get_random_data(self.image_path[index2], self.input_shape, random=self.train)
            
            # Generate mixup lambda
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # Blend images
            image = (image.astype(np.float32) * lam + image2.astype(np.float32) * (1 - lam))
            image = image.clip(0, 255).astype(np.uint8)
            # Combine boxes
            #box = np.concatenate([box, box2], axis=0) if len(box) + len(box2) > 0 else np.array([]) 
            if len(box) > 0 and len(box2) > 0:
                box = np.concatenate([box, box2], axis=0)
            elif len(box2) > 0:
                box = box2

        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        
        if len(box) != 0:
            boxes = np.array(box[:, :4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)

        for i in range(len(box)):
            bbox    = boxes[i].copy()
            cls_id  = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   计算真实框所属的特征点
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if self.center_sampling:
                    # Center sampling: merkez etrafındaki alanı pozitif olarak işaretle
                    r = self.center_sampling_radius
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            # Merkez etrafında grid içindeki noktaları kontrol et
                            cur_pt_x = ct_int[0] + dx
                            cur_pt_y = ct_int[1] + dy
                            
                            # Sınırları kontrol et
                            if (0 <= cur_pt_y < self.output_shape[0] and 
                                0 <= cur_pt_x < self.output_shape[1]):
                                
                                # Merkez noktadan uzaklığı hesapla
                                cur_pt = np.array([cur_pt_x, cur_pt_y], dtype=np.float32)
                                dist = np.sqrt(np.sum((ct - cur_pt)**2))
                                
                                # Gaussian çiz
                                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], [cur_pt_x, cur_pt_y], radius)
                                
                                # Merkez noktaya daha yakın olan noktalar için wh ve reg değerlerini güncelle
                                # Eğer daha önce bir değer atanmadıysa veya bu nokta merkeze daha yakınsa
                                if batch_reg_mask[cur_pt_y, cur_pt_x] == 0 or dist < np.sqrt(np.sum((ct - np.array([batch_reg[cur_pt_y, cur_pt_x, 0] + cur_pt_x, 
                                    batch_reg[cur_pt_y, cur_pt_x, 1] + cur_pt_y]))**2)):

                                    batch_wh[cur_pt_y, cur_pt_x] = 1. * w, 1. * h
                                    batch_reg[cur_pt_y, cur_pt_x] = ct - cur_pt
                                    batch_reg_mask[cur_pt_y, cur_pt_x] = 1
                else:
                    # Orijinal yaklaşım: sadece merkez noktayı işaretle
                    batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                    batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                    batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                    batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask
            
        
    def get_mosaic_data(self, index):
        """Mosaic augmentation: Combines 4 images into one."""
        indices = [index] + [np.random.randint(0, self.length) for _ in range(3)]
        images, all_boxes = [], []
        for idx in indices:
            img_path = self.image_path[idx]
            img, box = self.get_random_data(
                img_path,
                input_shape=(self.input_shape[0] // 2, self.input_shape[1] // 2),
                random=True,
                scale_ = (1,2)
            )
            images.append(img)
            all_boxes.append(box)

        # Create mosaic image
        mosaic_image = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        positions = [
            (0, 0),
            (self.input_shape[1] // 2, 0),
            (0, self.input_shape[0] // 2),
            (self.input_shape[1] // 2, self.input_shape[0] // 2)
        ]
        for i in range(4):
            img = images[i]
            x_offset, y_offset = positions[i]
            h, w = img.shape[0], img.shape[1]
            mosaic_image[y_offset:y_offset + h, x_offset:x_offset + w] = img

        # Combine and adjust boxes
        mosaic_boxes = []
        for i in range(4):
            boxes = all_boxes[i]
            if len(boxes) == 0:
                continue
            boxes = boxes.copy()
            x_off, y_off = positions[i]
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            mosaic_boxes.append(boxes)
        mosaic_boxes = np.concatenate(mosaic_boxes, axis=0) if len(mosaic_boxes) > 0 else np.array([])

        # Apply flip
        if self.rand() < 0.5:
            mosaic_image = mosaic_image[:, ::-1]
            if len(mosaic_boxes) > 0:
                mosaic_boxes[:, [0, 2]] = self.input_shape[1] - mosaic_boxes[:, [2, 0]]

        # Apply color jitter
        image_data = mosaic_image.astype(np.uint8)
        r = np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.4] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Filter boxes
        if len(mosaic_boxes) > 0:
            mosaic_boxes[:, 0:2] = np.clip(mosaic_boxes[:, 0:2], 0, self.input_shape[1])
            mosaic_boxes[:, 2] = np.clip(mosaic_boxes[:, 2], 0, self.input_shape[1])
            mosaic_boxes[:, 3] = np.clip(mosaic_boxes[:, 3], 0, self.input_shape[0])
            box_w = mosaic_boxes[:, 2] - mosaic_boxes[:, 0]
            box_h = mosaic_boxes[:, 3] - mosaic_boxes[:, 1]
            mosaic_boxes = mosaic_boxes[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, mosaic_boxes
    
    def bbox_areas_log_np(self,bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        return np.log(area) 


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image_path, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4,scale_ = (0.25,2), random=True):
        extension = os.path.splitext(image_path)[1]
        #line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(image_path)
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        if os.path.isfile(image_path.replace(extension,".xml")):
            annotation_line = image_path.replace(extension,".xml")
            box = extract_coordinates(annotation_line,self.classes)
        else:
            box = []
        #box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                box = np.array(box)
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        #scale = self.rand(.25, 2)
        scale = self.rand(scale_[0],scale_[1])
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            box = np.array(box)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box




# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs            = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    batch_hms       = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_whs       = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs      = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks

