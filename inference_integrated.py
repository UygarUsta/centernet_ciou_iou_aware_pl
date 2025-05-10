import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
import numpy as np 
import cv2 
import openvino as ov 
from torchvision.ops import nms
import time 
from dataloader import cvtColor
from PIL import Image 

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                    bias=bias),
                                    nn.BatchNorm2d(in_channels))
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias'):
            nn.init.constant_(module.bias, bias)


class ComplexUpsample(nn.Module):
    def __init__(self, input_dim=128, outpt_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv2 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, inputs):
        # do preprocess

        x = self.conv1(inputs)
        y = self.conv2(inputs)

        z = x + y

        z = nn.functional.interpolate(z, scale_factor=2,mode='bilinear' )

        return z
    

class Fpn(nn.Module):
    def __init__(self,input_dims=[24,32,96,320],head_dims=[128,128,128] ):
        super().__init__()





        self.latlayer2=nn.Sequential(SeparableConv2d(input_dims[0],head_dims[0]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[0]//2),
                                      nn.ReLU(inplace=True))


        self.latlayer3=nn.Sequential(SeparableConv2d(input_dims[1],head_dims[1]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[1]//2),
                                      nn.ReLU(inplace=True))

        self.latlayer4 = nn.Sequential(SeparableConv2d(input_dims[2], head_dims[2] // 2,kernel_size=5,padding=2),
                                       nn.BatchNorm2d(head_dims[2] // 2),
                                       nn.ReLU(inplace=True))



        self.upsample3=ComplexUpsample(head_dims[1],head_dims[0]//2)

        self.upsample4 =ComplexUpsample(head_dims[2],head_dims[1]//2)

        self.upsample5 = ComplexUpsample(input_dims[3],head_dims[2]//2)




    def forward(self, inputs):
        ##/24,32,96,320
        c2, c3, c4, c5 = inputs

        c4_lat = self.latlayer4(c4)
        c3_lat = self.latlayer3(c3)
        c2_lat = self.latlayer2(c2)


        upsample_c5=self.upsample5(c5)

        p4=torch.cat([c4_lat,upsample_c5],dim=1)


        upsample_p4=self.upsample4(p4)

        p3=torch.cat([c3_lat,upsample_p4],dim=1)

        upsample_p3 = self.upsample3(p3)

        p2 = torch.cat([c2_lat, upsample_p3],dim=1)


        return p2

class CenterNetHead(nn.Module):
    def __init__(self,nc,head_dims ):
        super().__init__()



        self.cls =SeparableConv2d(head_dims[0], nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh =SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset =SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.iou_head = nn.Conv2d(head_dims[0], 1, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls.pointwise, 0, 0.01,-2.19)
        normal_init(self.wh.pointwise, 0, 0.01, 0)



    def forward(self, inputs):


        cls = self.cls(inputs).sigmoid_()
        wh = self.wh(inputs)
        offset = self.offset(inputs)
        iou_aware_head = self.iou_head(inputs).sigmoid_().squeeze(1)
        return cls,wh,offset,iou_aware_head

class CenterNet(nn.Module):
    def __init__(self,nc,inference):
        super().__init__()

        self.nc = nc 
        self.inference = inference
        input_dims = []
        ###model structure
        self.backbone =  timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, features_only=True,exportable=True) #e1200_r224_in1k - 050
        feature_info = self.backbone.feature_info
        for idx, info in enumerate(feature_info.info):
            #print(f"Seviye {idx+1}: {info['module']}, kanal sayısı={info['num_chs']}")
            if idx > 0:
                input_dims.append(info['num_chs'])
        self.fpn=Fpn(input_dims=input_dims,head_dims=[128,128,128])
        self.head = CenterNetHead(self.nc,head_dims=[128,128,128])


        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    def forward(self, inputs):

        fms = self.backbone(inputs)[-4:]
        fpn_fm=self.fpn(fms)
        cls, wh, offset, iou_aware_head= self.head(fpn_fm)
        if not hasattr(self, 'inference') or not self.inference:
            return cls, wh, offset, iou_aware_head
        else:
            detections = self.decode(cls, wh, offset, iou_aware_head, stride=4)
            return detections


        
    
    def decode(self, heatmap, wh, offset, iou_aware=None, stride=4, K=100):
        def nms(heat, kernel=3):
            ##fast
            heat = heat.permute([0, 2, 3, 1])
            heat, clses = torch.max(heat, dim=3)

            heat = heat.unsqueeze(1)
            scores = heat #torch.sigmoid(heat)

            hmax = nn.MaxPool2d(kernel, 1, padding=1)(scores)
            keep = (scores == hmax).float()

            return scores * keep, clses
        
        def get_bboxes(wh, offset):
            ### decode the box with offset
            shifts_x = torch.arange(0, W, dtype=torch.float32, device=self.device)
            shifts_y = torch.arange(0, H, dtype=torch.float32, device=self.device)

            y_range, x_range = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            # Apply offset to center positions
            x_center = x_range + offset[:, 0, :, :].squeeze(0)
            y_center = y_range + offset[:, 1, :, :].squeeze(0)
            
            # Convert centers and sizes to box coordinates
            half_w = wh[:, 0, :, :] / 2
            half_h = wh[:, 1, :, :] / 2
            
            x1 = (x_center - half_w) * stride
            y1 = (y_center - half_h) * stride
            x2 = (x_center + half_w) * stride
            y2 = (y_center + half_h) * stride
            
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            return boxes

        batch, cat, H, W = heatmap.size()

        score_map, label_map = nms(heatmap)
        pred_boxes = get_bboxes(wh, offset)

        # Apply IoU-aware score if available
        if iou_aware is not None:
            iou_scores = torch.sigmoid(iou_aware)[None]
            # Reshape to match score_map
            iou_scores = iou_scores.permute([0, 2, 3, 1]).squeeze(-1).unsqueeze(1)
            score_map = score_map * iou_scores

        score_map = torch.reshape(score_map, shape=[batch, -1])
        
        # Get top K detections
        top_score, top_index = torch.topk(score_map, k=K)
        top_score = torch.unsqueeze(top_score, 2)
        
        # Gather boxes, labels for top K detections
        pred_boxes = torch.reshape(pred_boxes, shape=[batch, 4, -1])
        pred_boxes = pred_boxes.permute([0, 2, 1])
        
        # Regular indexing (assuming batch size 1 for simplicity)
        pred_boxes = pred_boxes[:, top_index[0], :]
        
        label_map = torch.reshape(label_map, shape=[batch, -1])
        label_map = label_map[:, top_index[0]]
        label_map = torch.unsqueeze(label_map, 2)

        # Convert to float for consistency
        pred_boxes = pred_boxes.float()
        label_map = label_map.float()
        
        # Concatenate boxes, scores, and labels
        detections = torch.cat([pred_boxes, top_score, label_map], dim=2)
        
        return detections
    
def load_model(model,model_path):
    device = "cuda"
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std


def resize_numpy(image, size):
    image = np.array(image,dtype='float32')
    w, h = size
    new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return new_image

def preprocess(image):
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
    else:
        img_bgr = image
    
    # Resize image without letterboxing for speed
    image = resize_numpy(img_bgr, (input_width, input_height))
    #image_ = image.copy()  # For visualization
    image = preprocess_input(image)

    # Convert to torch tensor (C, H, W) on GPU - single batch
    image_tensor = torch.tensor(image, dtype=torch.float32).to(DEVICE).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor

def scale_to_original(final_dets,img_bgr): 
    #pass image itself
    # Scale detections back to original image size
    bboxes = []
    for det in final_dets:
        xmin = int(det[0]) * img_bgr.shape[1] // input_width
        ymin = int(det[1]) * img_bgr.shape[0] // input_height
        xmax = int(det[2]) * img_bgr.shape[1] // input_width
        ymax = int(det[3]) * img_bgr.shape[0] // input_height
        score = float(det[4])
        cls_id = int(det[5])
        bboxes.append([xmin, ymin, xmax, ymax, score, cls_id])
    
    return bboxes

def scale_detections_to_original(detections, model_input_size, original_image_size):
    """
    Scale detection coordinates from model input size to original image size.
    
    Args:
        detections: Tensor of shape [batch, K, 6] with format [x1, y1, x2, y2, score, class_id]
        model_input_size: Tuple (height, width) of the model input size
        original_image_size: Tuple (height, width) of the original image
    
    Returns:
        List of scaled detections in format [[x1, y1, x2, y2, score, class_id], ...]
    """
    input_height, input_width = model_input_size
    orig_height, orig_width = original_image_size
    
    # Create scaling factors
    scale_w = orig_width / input_width
    scale_h = orig_height / input_height
    
    # Convert to numpy for easier handling (assuming batch size 1)
    detections_np = detections[0].cpu().numpy()
    
    # Create list to store scaled detections
    scaled_detections = []
    
    # Process each detection
    for det in detections_np:
        # Scale the bounding box coordinates
        x1 = int(det[0] * scale_w)
        y1 = int(det[1] * scale_h)
        x2 = int(det[2] * scale_w)
        y2 = int(det[3] * scale_h)
        score = float(det[4])
        cls_id = int(det[5])
        
        scaled_detections.append([x1, y1, x2, y2, score, cls_id])
    
    return scaled_detections

if __name__ == "__main__":

    input_height = 512
    input_width = 512
    confidence = 0.3

    f = open("classes.txt","r").readlines()
    classes = []
    for i in f:
        classes.append(i.strip('\n'))

    print(classes)

    trace = True

    DEVICE = "cuda"
    model = CenterNet(nc=len(classes),inference=True)
    device = "cuda"
    model_path = "./coco_mbv4_ciou_aware_best_model_mAP_0.2332.pth"
    model = load_model(model,model_path).to(DEVICE).eval()

    if trace:
        dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
        print("Start Tracing")
        model = torch.jit.trace(model, dummy_input)
        print("End Tracing")

    video_path = "/home/rivian/Desktop/videos/0010.mp4"
    cap = cv2.VideoCapture(video_path)
    while 1:
        ret,image = cap.read()
        image_copy = image.copy()
        fps_start = time.time()
        image = image[...,::-1]
        image_preprocessed = preprocess(image) #returns tensor

        fps_start = time.time()
        with torch.no_grad():
            output = model(image_preprocessed)
        fps_end = time.time()
        
        dets = scale_detections_to_original(output,(input_height,input_width),(image.shape[0],image.shape[1]))
        filtered_dets = [det for det in dets if det[4] > confidence]
        for det in filtered_dets:
            xmin,ymin,xmax,ymax,score,cls_id = det
            cv2.rectangle(image_copy,(xmin,ymin),(xmax,ymax),(0,255,0),3)
            cv2.putText(image_copy,classes[cls_id],(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(image_copy,f'{score:.2f}',(xmax-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        
        fps = 1/(fps_end-fps_start)
        cv2.putText(image_copy,f"Model FPS: {fps:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("image",image_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


