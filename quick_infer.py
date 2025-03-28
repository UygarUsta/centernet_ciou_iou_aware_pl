import torch 
import numpy as np 
import cv2 
import openvino as ov 
from torchvision.ops import nms
import torch.nn as nn 
import time 

input_width = 224
input_height = 224

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
from hardnet import get_pose_net
device = "cuda"
conf = 0.4
model_path = "best_epoch_weights.pth"
model = get_pose_net(85,{"hm":len(classes),"wh":4})
model = load_model(model,model_path).to(DEVICE).eval()

if trace:
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")
    #model.save(f"{model_path.split('/')[-1].split('.')[0] + '_traced.pth'}")

dummy_input = torch.randn(1, 3, input_height, input_width).to(DEVICE)
model =  ov.compile_model(ov.convert_model(model, example_input=dummy_input))


def inference_video(image, classes, stride=4, confidence=0.45, nms_iou=0.3):
    """
    Inference on a single image using a CenterNet-style approach
    with a [left, top, right, bottom] offset scheme.

    Args:
        image_path (str) : Path to the image file
        classes (list)   : List of class names
        stride (int)     : Downsample stride between input and feature maps
        confidence (float): Threshold to filter out low-confidence center points
        nms_iou (float)  : IoU threshold for NMS

    Returns:
        pred_reg_mask (tensor): The regression offsets that pass threshold (debugging)
    """
    #fps1 = time.time()
    # Read and preprocess the image
    if type(image) == str:
        img_bgr = cv2.imread(image)
    else:
        img_bgr = image
    image   = resize_numpy(img_bgr, (input_width, input_height),letterbox_image=False) #cv2.resize(img_bgr, (input_width, input_height))
    image_  = image.copy()  # For visualization
    image   = preprocess_input(image)

    # Convert to torch tensor (C, H, W) on GPU
    image_tensor = torch.tensor(image, dtype=torch.float32).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
    #fpre = time.time()
    #print(f"Preprocessing took: {fpre - fps1} ms")
    #f1 = time.time()
    with torch.no_grad():
        pred = model(image_tensor)  # forward pass
    #f2 = time.time()
    #print(f"Model inference time: {f2-f1} ms")
    #fp1 = time.time()
    hm =  pred[0]  # Heatmap and Regression outputs
    reg = pred[1]
    hm = torch.tensor(hm)
    reg = torch.tensor(reg)
    
    #hm_dist = hm[:,-1,:,:].unsqueeze(0)
    #hm = hm[:,:-1,:,:]
    #hm = torch.sigmoid(hm)  # apply sigmoid to heatmap
    
    # Convert heatmap to numpy just for visualization of the first channel
    #hm_np = hm.permute(0, 2, 3, 1).numpy()
    #hm_np = np.squeeze(hm_np, 0)  # shape: (H, W, num_classes)

    # pool_nms is presumably your local-maxima function on the heatmap
    # Make sure you apply it on 'hm' (PyTorch tensor) not 'hm_np'
    hm = pool_nms(hm)  # shape: (B, num_classes, H, W)

    b, c, output_h, output_w = hm.shape
    detects = []

    for batch_i in range(b):
        # Flatten heatmap and regression for easier indexing
        heat_map  = hm[batch_i].permute(1, 2, 0).view(-1, c)        # (H*W, num_classes)
        pred_reg_ = reg[batch_i].permute(1, 2, 0).view(-1, 4)       # (H*W, 4)
        #pred_dist = hm_dist[batch_i].permute(1, 2, 0).view(-1, 1)
        # Generate a grid of indices to locate each pixel in feature space
        yv, xv = torch.meshgrid(
            torch.arange(0, output_h, dtype=torch.float32),
            torch.arange(0, output_w, dtype=torch.float32)
        )
        xv = xv.flatten() #.cuda()  # (H*W,)
        yv = yv.flatten() #.cuda()  # (H*W,)

        # Find the class with the highest score at each pixel
        class_conf, class_pred = torch.max(heat_map, dim=-1)  # (H*W,) each

        # Apply confidence threshold
        mask = class_conf > confidence

        # Filter out only the pixels above the threshold
        pred_reg_mask = pred_reg_[mask]  # shape: (M, 4)
        #pred_dist_mask = pred_dist[mask]
        scores_mask    = class_conf[mask]
        classes_mask   = class_pred[mask]

        if len(pred_reg_mask) == 0:
                detects.append([])
                continue  

        # Decode bounding boxes from offsets
        # According to your dataset code:
        # batch_reg[..., 0] = left_offset
        # batch_reg[..., 1] = top_offset
        # batch_reg[..., 2] = right_offset
        # batch_reg[..., 3] = bottom_offset
        left_offset   = pred_reg_mask[:, 0]
        top_offset    = pred_reg_mask[:, 1]
        right_offset  = pred_reg_mask[:, 2]
        bottom_offset = pred_reg_mask[:, 3]

        # Convert center coords from feature space -> input space
        x_center = xv[mask] * stride
        y_center = yv[mask] * stride

        # Now compute x_min, y_min, x_max, y_max
        x_min = x_center - left_offset   * stride
        y_min = y_center - top_offset    * stride
        x_max = x_center + right_offset  * stride
        y_max = y_center + bottom_offset * stride
        
        #distance = pred_dist_mask[:,0].float().unsqueeze(-1)

        # Concatenate for NMS: [x_min, y_min, x_max, y_max, score, class_id]
        bboxes  = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        scores  = scores_mask.unsqueeze(-1)
        c_ids   = classes_mask.float().unsqueeze(-1)

        detect = torch.cat([bboxes, scores, c_ids], dim=-1)  # shape: (M, 6)
        detects.append(detect)

    # For single-batch inference, we only look at detects[0]
    # Format: [x_min, y_min, x_max, y_max, score, class_id]
    all_detections = detects[0]
    if len(all_detections) > 0:
        # NMS on CPU or GPU
        keep_indices = nms(all_detections[:, :4], all_detections[:, 4], nms_iou)
        final_dets   = all_detections[keep_indices]
    else:
        final_dets = all_detections

    #fp2 = time.time()
    #print(f"Postprocessing took: {fp2-fp1} ms")
    # Draw results
    bboxes = []
    for det in final_dets:
        xmin = int(det[0]) * img_bgr.shape[1] // input_width
        ymin = int(det[1]) * img_bgr.shape[0] // input_height
        xmax = int(det[2]) * img_bgr.shape[1] // input_width
        ymax = int(det[3]) * img_bgr.shape[0] // input_height
        score = float(det[4])
        cls_id = int(det[5])
        #distance = float(det[6])
        bboxes.append([xmin,ymin,xmax,ymax,score,cls_id])

        # cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(
        #     image_,
        #     f"{classes[cls_id]}: {score:.2f} d:{distance}",
        #     (xmin, ymin - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     1,
        # )

    # Show the first channel of the heatmap (for debugging)
    # plt.imshow(cv2.resize(hm_np[:, :, 0], (512, 512)))
    # plt.title("Heatmap - First Class Channel")
    # plt.show()

    # # Show the detection results
    # plt.imshow(cv2.cvtColor(image_, cv2.COLOR_BGR2RGB))
    # plt.title("Detections")
    # plt.show()

    return bboxes,image_

video_path = "/home/rivian/Desktop/0010.mp4"
cap = cv2.VideoCapture(video_path)
while 1:
    ret,image = cap.read()
    image_copy = image.copy()
    fps_start = time.time()
    image = image[...,::-1]
    bboxes,image_anotated = inference_video(image,classes)
    fps_end = time.time()
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

