import torch 
import numpy as np 
import cv2 
import openvino as ov 
from torchvision.ops import nms
import torch.nn as nn 
import time 
from dataloader import cvtColor
from PIL import Image 

def pool_nms(heat, kernel=3):
    """Efficient max pooling-based NMS."""
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_bbox_fast(pred_hms, pred_whs, pred_offsets, pred_ious=None, confidence=0.3, cuda=True):
    """Faster implementation of bbox decoding."""
    # Apply non-maximum suppression to heatmaps
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    
    # Create coordinate grid once (shared across batch)
    yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
    xv, yv = xv.flatten().float(), yv.flatten().float()
    if cuda:
        xv = xv.cuda()
        yv = yv.cuda()
    
    # Process each image in batch
    for batch in range(b):
        # Reshape tensors for efficient processing
        heat_map = pred_hms[batch].permute(1, 2, 0).reshape(-1, c)
        pred_wh = pred_whs[batch].permute(1, 2, 0).reshape(-1, 2)
        pred_offset = pred_offsets[batch].permute(1, 2, 0).reshape(-1, 2)
        
        # Get class confidence and predictions
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence
        
        # Skip if no detections pass confidence threshold
        if not mask.any():
            detects.append([])
            continue
        
        # Get IoU map if available
        if pred_ious is not None:
            iou_map = pred_ious[batch]
            if iou_map.dim() == 3 and iou_map.shape[0] == 1:
                iou_map = iou_map.squeeze(0)
            iou_map = iou_map.reshape(-1)
            iou_mask = iou_map[mask]
        else:
            iou_mask = None
        
        # Filter by mask
        pred_wh_mask = pred_wh[mask]
        pred_offset_mask = pred_offset[mask]
        xv_mask = xv[mask] + pred_offset_mask[..., 0]
        yv_mask = yv[mask] + pred_offset_mask[..., 1]
        
        # Calculate bbox coordinates (vectorized)
        half_w, half_h = pred_wh_mask[:, 0:1] / 2, pred_wh_mask[:, 1:2] / 2
        
        # Create bboxes directly without unnecessary concatenation
        bboxes = torch.zeros((xv_mask.size(0), 4), device=xv_mask.device)
        bboxes[:, 0] = (xv_mask - half_w.squeeze(-1)) / output_w
        bboxes[:, 1] = (yv_mask - half_h.squeeze(-1)) / output_h
        bboxes[:, 2] = (xv_mask + half_w.squeeze(-1)) / output_w
        bboxes[:, 3] = (yv_mask + half_h.squeeze(-1)) / output_h
        
        # Calculate final confidence
        if iou_mask is not None:
            final_conf = class_conf[mask] * iou_mask
        else:
            final_conf = class_conf[mask]
            
        # Create detection results
        detect = torch.cat([
            bboxes,
            final_conf.unsqueeze(-1),
            class_pred[mask].float().unsqueeze(-1)
        ], dim=-1)
        
        detects.append(detect)
    
    return detects

def centernet_correct_boxes_fast(boxes, input_shape, image_shape, letterbox_image):
    """Optimized version of box correction."""
    # Convert to numpy arrays for consistency with original function
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)
    
    # Extract coordinates
    x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    
    # Calculate centers and dimensions
    box_xy = np.concatenate([(x1 + x2) / 2, (y1 + y2) / 2], axis=-1)
    box_wh = np.concatenate([x2 - x1, y2 - y1], axis=-1)
    
    # Switch to yx format for processing
    box_yx = box_xy[:, ::-1]
    box_hw = box_wh[:, ::-1]
    
    if letterbox_image:
        # Calculate letterbox adjustments
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        
        # Apply letterbox adjustments
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
    
    # Calculate box corners
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    
    # Format boxes and scale to image dimensions
    boxes = np.concatenate([
        box_mins[:, 0:1], box_mins[:, 1:2],
        box_maxes[:, 0:1], box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    
    return boxes

def postprocess_fast(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    """Faster implementation of postprocessing."""
    batch_size = len(prediction)
    output = [None] * batch_size
    
    for i, detections in enumerate(prediction):
        if len(detections) == 0:
            continue
            
        # Process on GPU when possible
        if isinstance(detections, torch.Tensor):
            is_cuda = detections.is_cuda
            unique_labels = detections[:, -1].cpu().unique()
            if is_cuda:
                unique_labels = unique_labels.cuda()
                
            # Process each class
            for c in unique_labels:
                # Get detections for this class
                detections_class = detections[detections[:, -1] == c]
                
                if need_nms:
                    # Use torchvision's NMS which is faster
                    keep = nms(
                        detections_class[:, :4],
                        detections_class[:, 4],
                        nms_thres
                    )
                    max_detections = detections_class[keep]
                else:
                    max_detections = detections_class
                
                # Add to output
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
        
        # Convert to numpy and apply box correction
        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            
            # Apply box correction
            output[i][:, :4] = centernet_correct_boxes_fast(
                output[i][:, :4], 
                input_shape, 
                image_shape, 
                letterbox_image
            )
    
    return output



class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors() 

input_width = 512
input_height = 512

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

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


def hardnet_load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model
  
f = open("classes.txt","r").readlines()
classes = []
for i in f:
    classes.append(i.strip('\n'))

print(classes)

trace = False

DEVICE = "cpu"
from mbv4_timm import CenterNet
model = CenterNet(nc=len(classes))
device = "cpu"
model_path = "/home/rivian/Desktop/centernet_ciou_iou_aware_pl/lightning_logs/centernet/version_0/checkpoints/best_model_mAP_0.4819.pth"
model = load_model(model,model_path).to(DEVICE).eval()



if trace:
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")
    #model.save(f"{model_path.split('/')[-1].split('.')[0] + '_traced.pth'}")

dummy_input = torch.randn(1, 3, input_height, input_width).to(DEVICE)
model =  ov.compile_model(ov.convert_model(model, example_input=dummy_input))



def infer_image(model, img, classes, stride=4, confidence=0.35, half=False, input_shape=(512, 512), cpu=False, openvino_exp=False):
    # Device setup - moved outside function if possible
    device = torch.device("cpu" if cpu else "cuda")
    cuda = not cpu
    
    # Start timing
    #fps1 = time.time()
    
    # More efficient image handling
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = img
    
    # Cache image shape once
    image_shape = np.array(image.shape[:2] if hasattr(image, 'shape') else image.size[::-1])
    
    # Convert image format efficiently
    image = cvtColor(image)
    
    # Use faster resize method
    image_data = resize_numpy(image, input_shape, letterbox_image=False)
    
    # Optimize preprocessing with vectorized operations
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    image = np.array(image)
    # Calculate rectangle parameters once
    lf = max(round(sum(image_shape) / 2 * 0.003), 2)
    tf = max(lf - 1, 1)
    
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    box_annos = []
    try:
        # Model inference with optimized memory usage
        with torch.no_grad():
            # Convert to tensor once and reuse
            images = torch.from_numpy(image_data).to(device, dtype=torch.float16 if half else torch.float32)
            
            # Run inference
            if not openvino_exp:
                hm, wh, offset, iou_pred = model(images)
            else:
                output = model(images)
                hm = torch.tensor(output[0])
                wh = torch.tensor(output[1])
                offset = torch.tensor(output[2])
                iou_pred = torch.tensor(output[3])
        pf1 = time.time()
        # Decode bounding boxes
        outputs = decode_bbox_fast(hm, wh, offset, iou_pred, confidence=confidence, cuda=cuda)
        
        # Post-processing
        results = postprocess_fast(outputs, True, image_shape, input_shape, False, 0.3)
        pf2 = time.time()
        print(f"Postprocessing time is: {pf2-pf1}")
        
        # Batch drawing operations for better performance
        if results[0] is not None and len(results[0]) > 0:
            for det in results[0]:
                xmin, ymin = int(det[1]), int(det[0])
                xmax, ymax = int(det[3]), int(det[2])
                conf = float(det[4])
                class_label = int(det[5])
                box = [ymin, xmin, ymax, xmax]
                name = f'{classes[class_label]} {conf:.2f}'
                box_annos.append([xmin, ymin, xmax, ymax, name, conf])
                
                # Draw rectangle
                color = colors(class_label, True)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, lf)
                
                # Draw text more efficiently
                cv2.putText(image, name, (xmin-3, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                
    except Exception as e:
        print(f"Exception: {e}")
    
    # Calculate FPS
    #fps2 = time.time()
    #fps = 1 / (fps2 - fps1)
    #cv2.putText(image, f'FPS:{fps:.2f}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    
    return image, box_annos



def inference_video_ciou(image, classes, stride=4, confidence=0.45, nms_iou=0.3,use_iou=True):
    """
    Optimized inference for CenterNet with CIoU awareness.
    
    Args:
        image (str or ndarray): Path to image or image array
        classes (list): List of class names
        stride (int): Downsample stride between input and feature maps
        confidence (float): Threshold to filter out low-confidence center points
        nms_iou (float): IoU threshold for NMS
        
    Returns:
        bboxes (list): List of [xmin, ymin, xmax, ymax, score, cls_id]
        image_ (ndarray): Copy of input image for visualization
    """
    # Read and preprocess the image
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
    else:
        img_bgr = image
    
    # Resize image without letterboxing for speed
    image = resize_numpy(img_bgr, (input_width, input_height), letterbox_image=False)
    image_ = image.copy()  # For visualization
    image = preprocess_input(image)

    # Convert to torch tensor (C, H, W) on GPU - single batch
    image_tensor = torch.tensor(image, dtype=torch.float32).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
    
    # Model inference
    with torch.no_grad():
        pred = model(image_tensor)
    
    # Extract predictions
    pred_hms = torch.tensor(pred[0])  # Heatmap outputs
    pred_whs = torch.tensor(pred[1])  # Width/Height outputs
    pred_offsets = torch.tensor(pred[2])  # Offset outputs
    pred_ious = torch.tensor(pred[3]) #[None]  # IoU awareness outputs

    
    # Apply sigmoid to heatmap if not already applied in model
    # pred_hms = torch.sigmoid(pred_hms)
    
    # Apply pool NMS to get local maxima
    pred_hms = pool_nms(pred_hms)
    
    # Get output dimensions
    b, c, output_h, output_w = pred_hms.shape
    
    # Generate coordinate grid once (more efficient)
    yv, xv = torch.meshgrid(
        torch.arange(0, output_h, dtype=torch.float32, device=DEVICE),
        torch.arange(0, output_w, dtype=torch.float32, device=DEVICE)
    )
    xv = xv.flatten()  # (H*W,)
    yv = yv.flatten()  # (H*W,)
    
    # Process batch (usually just one image)
    detects = []
    
    for batch_i in range(b):
        # Flatten predictions for easier processing
        heat_map = pred_hms[batch_i].permute(1, 2, 0).reshape(-1, c)  # (H*W, num_classes)
        pred_wh = pred_whs[batch_i].permute(1, 2, 0).reshape(-1, 2)  # (H*W, 2)
        pred_offset = pred_offsets[batch_i].permute(1, 2, 0).reshape(-1, 2)  # (H*W, 2)
        
        # Extract IoU awareness - handle different possible shapes
        if pred_ious.dim() == 4 and use_iou:
            # Shape [B, 1, H, W] or [B, C, H, W]
            iou_map = pred_ious[batch_i]
            if iou_map.dim() == 3 and iou_map.shape[0] == 1:
                iou_map = iou_map.squeeze(0)  # [H, W]
            elif iou_map.dim() == 3:
                # Multiple IoU maps (one per class)
                iou_map = iou_map.permute(1, 2, 0)  # [H, W, C]
            iou_map = iou_map.reshape(-1) if iou_map.dim() == 2 else iou_map.reshape(-1, c)
        else:
            iou_map = None
        
        # Find the class with highest confidence at each point
        class_conf, class_pred = torch.max(heat_map, dim=-1)  # (H*W,) each
        
        # Apply confidence threshold
        mask = class_conf > confidence
        
        # Skip if no detections pass threshold
        if not mask.any():
            detects.append([])
            continue
        # Filter predictions by mask
        pred_wh_mask = pred_wh[mask]  # (N, 2)
        pred_offset_mask = pred_offset[mask]  # (N, 2)
        # Get IoU-aware confidence if available
        if iou_map is not None:
            if iou_map.dim() == 1:
                # Single IoU map
                iou_mask = iou_map[mask]  # (N,)
                final_conf = class_conf[mask] * iou_mask
            else:
                # Per-class IoU maps
                class_indices = class_pred[mask].unsqueeze(1).long()  # (N, 1)
                iou_mask = torch.gather(iou_map[mask], 1, class_indices).squeeze(1)  # (N,)
                final_conf = class_conf[mask] * iou_mask
        else:
            final_conf = class_conf[mask]
        
        # Get center coordinates from feature map
        xv_mask = xv[mask] + pred_offset_mask[..., 0]  # (N,)
        yv_mask = yv[mask] + pred_offset_mask[..., 1]  # (N,)
        
        # Calculate bbox dimensions
        half_w, half_h = pred_wh_mask[..., 0] / 2, pred_wh_mask[..., 1] / 2
        
        # Calculate bbox coordinates directly (vectorized)
        x_min = (xv_mask - half_w) * stride
        y_min = (yv_mask - half_h) * stride
        x_max = (xv_mask + half_w) * stride
        y_max = (yv_mask + half_h) * stride
        
        # Stack bbox coordinates
        bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (N, 4)
        
        # Create detection tensor with all info
        detect = torch.cat([
            bboxes,
            final_conf.unsqueeze(-1),
            class_pred[mask].float().unsqueeze(-1)
        ], dim=-1)  # (N, 6)
        
        detects.append(detect)
    
    # Process detections (for single batch)
    all_detections = detects[0]
    
    # Apply NMS if we have any detections
    if len(all_detections) > 0:
        # Use torchvision's optimized NMS
        keep_indices = nms(all_detections[:, :4], all_detections[:, 4], nms_iou)
        final_dets = all_detections[keep_indices]
    else:
        final_dets = all_detections
    
    # Convert to original image coordinates
    bboxes = []
    for det in final_dets:
        xmin = int(det[0]) * img_bgr.shape[1] // input_width
        ymin = int(det[1]) * img_bgr.shape[0] // input_height
        xmax = int(det[2]) * img_bgr.shape[1] // input_width
        ymax = int(det[3]) * img_bgr.shape[0] // input_height
        score = float(det[4])
        cls_id = int(det[5])
        bboxes.append([xmin, ymin, xmax, ymax, score, cls_id])
    
    return bboxes, image_

video_path = "/home/rivian/Desktop/0010.mp4"
cap = cv2.VideoCapture(video_path)
while 1:
    ret,image = cap.read()
    image_copy = image.copy()
    fps_start = time.time()
    image = image[...,::-1]
    bboxes,image_anotated = inference_video_ciou(image, classes) #infer_image(model, image, classes)
    fps_end = time.time()
    image_anotated = np.array(image_anotated)
    
    image = image[...,::-1]
    for det in bboxes:
        xmin = np.clip(int(det[0]),0,image.shape[1]) 
        ymin = np.clip(int(det[1]),0,image.shape[0])  
        xmax = np.clip(int(det[2]),0,image.shape[1])  
        ymax = np.clip(int(det[3]),0,image.shape[0]) 
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,255),2)
        cv2.putText(image,f"{classes[int(det[5])]}:{det[4]:.2f}",(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    fps = 1/(fps_end-fps_start)
    cv2.putText(image,f"FPS:{fps:.2f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imshow("image",image)
    ch = cv2.waitKey(1)
    if ch == ord("q"):
       break

