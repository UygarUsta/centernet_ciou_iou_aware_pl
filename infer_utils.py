import torch 
import numpy as np
from dataloader import preprocess_input,resize_image,cvtColor,resize_numpy,preprocess_input_simple
from utils_bbox import decode_bbox,postprocess,centernet_correct_boxes_xyxy
from PIL import Image 
import cv2 
import time 


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


def infer_image(model,img,classes,stride=4,confidence=0.05,half=False,input_shape = (512,512),cpu = False,openvino_exp=False):
    #<class_name> <confidence> <left> <top> <right> <bottom>
    #files = glob(folder + "val_images/*.jpg") + glob(folder + "val_images/*.png")
    if cpu:
       device = torch.device("cpu")
       cuda = False 
    else:
       device = torch.device("cuda")
       cuda = True
       
    fps1 = time.time()
    if type(img) == str:
        image =  Image.open(img) 
    else:
        image = img #faster
        #image = Image.fromarray(img)
    
    image_shape = np.array(np.shape(image)[0:2])
    image  = cvtColor(image)
    #image_data = resize_image(image,tuple(input_shape),letterbox_image=True) 
    image_data = resize_numpy(image,tuple(input_shape),letterbox_image=False)
    #image_data = cv2.resize(image, tuple(input_shape), interpolation=cv2.INTER_CUBIC)

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    #image_data = np.expand_dims(np.transpose(preprocess_input_simple(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    lf = max(round(sum(image_shape) / 2 * 0.003), 2) #rectangle thickness
    tf = max(lf - 1, 1)  # font thickness

    image = np.array(image)
    #fpre = time.time()
    #print(f"Preprocessing took: {fpre - fps1} ms")
    box_annos = []
    try:
      #f1 = time.time()
      with torch.no_grad():
          images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor).to(device)
          if half: images = images.half()
          if not openvino_exp:
            hm,wh,offset,iou_pred = model(images)
          if openvino_exp:
              output = model(images) # hm,wh,offset
              hm = torch.tensor(output[0])
              wh = torch.tensor(output[1])
              offset = torch.tensor(output[2])
              iou_pred = torch.tensor(output[3])
      if half: 
          hm  = hm.half()
          wh = wh.half()
          offset = offset.half()
          iou_pred = iou_pred.half()
      #f2 = time.time()
      #print(f"Model inference time: {f2-f1} ms")

      #fp1 = time.time()
      outputs = decode_bbox(hm,wh,offset,iou_pred,confidence=confidence,cuda=cuda)
      
      #results = centernet_correct_boxes_xyxy(outputs,input_shape, image_shape, False).cpu()
      #results = outputs[0].cpu().numpy()
      results = postprocess(outputs,True,image_shape,input_shape, False, 0.3) #letterbox true

      #fp2 = time.time()
      #print(f"Postprocessing took: {fp2-fp1} ms")

      for det in results[0]: #Batch size1
        xmin = int(det[1])
        ymin = int(det[0])
        xmax = int(det[3])
        ymax = int(det[2])
        conf = float(det[4])
        label = int(det[5])
        class_label = label
        box = [ymin,xmin,ymax,xmax]
        name = f'{classes[class_label]} {conf:.2f}'
        box_annos.append([xmin,ymin,xmax,ymax,str(name),conf])

        p1, p2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
        w, h = cv2.getTextSize(name, 0, fontScale=lf / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        #cv2.rectangle(image, p1, p2, colors(class_label, True), -1, cv2.LINE_AA)  # filled

        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),colors(class_label, True),lf) #(0,255,0)
        #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),colors(class_label, True),1) #(0,255,0)
        #cv2.putText(image,str(name),(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),tf) #(255,0,255)
        #cv2.putText(image,str(conf),(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),tf)
        cv2.putText(image,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

            
    except Exception as e:
        print("Excepton:",e)
        pass
    fps2 = time.time()
    fps = 1 / (fps2-fps1) 
    cv2.putText(image,f'FPS:{fps:.2f}',(200,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        #print(f"Could not infer an error occured: {e}")

    return image,box_annos

            
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
