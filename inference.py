import torch
import numpy as np
import cv2
from infer_utils import infer_image,load_model
from glob import glob
import os
import time 

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

video = False
half = False 
cpu = False 
trace = False 
openvino_exp = False 
openvino_int8 = False 
export_onnx = False 



f = open("classes.txt","r").readlines()
classes = []
for i in f:
    classes.append(i.strip('\n'))

print(classes)

input_height = 512
input_width = 512
stride = 4
folder =  "/home/rivian/Desktop/Datasets/derpet_v4_label_tf/val_images" 
video_path = "G:/vlc-record-2025-01-22-11h07m03s-2_10.34.09_novis_output.avi-.avi" 
model_path = "/home/rivian/Desktop/centernet_ciou_iou_aware_pl/lightning_logs/centernet/version_1/checkpoints/best_model_mAP_0.4318.pth"
device = "cuda"
model_type = "mbv4_timm"

if model_type == "mbv4_timm":
    conf = 0.35
    from mbv4_timm import CenterNet
    model = CenterNet(nc=len(classes))
    if model_path != "":
        model = load_model(model,model_path)



model.cuda()
#model = torch.compile(model) #experimental

model.eval()


if cpu:
    model.cpu()
    device = torch.device("cpu")

if half:
    model.half()

if trace:
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")
    model.save(f"{model_path.split('/')[-1].split('.')[0] + '_traced.pth'}")

if openvino_exp:
    import openvino as ov
    import random 
    core = ov.Core()
    if openvino_int8:
        def worker_init_fn(worker_id, rank, seed):
            worker_seed = rank + seed
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        print("Start Profiling")
        from dataloader import CenternetDataset,centernet_dataset_collate
        from torch.utils.data import DataLoader
        from functools import partial
        import nncf 
        input_shape = (input_width,input_height)
        batch_size = 8
        num_workers = 4 
        rank = 0
        seed = 11
        val_sampler = None
        def transform_fn(data_item):
            output = data_item
            return output[0].float()
        folder_dl = "/home/rivian/Desktop/Datasets/derpetv5_xml"
        val_images = glob(os.path.join(folder_dl,"val_images","*.jpg")) + glob(os.path.join(folder_dl,"val_images","*.png")) + glob(os.path.join(folder_dl,"val_images","*.JPG"))
        val_dataset = CenternetDataset(val_images,input_shape,classes,len(classes),train=False)
        gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        calibration_dataset = nncf.Dataset(gen_val, transform_fn)
        quantized_model = nncf.quantize(model.cpu().float(), calibration_dataset)

        dummy_input = torch.randn(1, 3, input_width,input_height).float()
        quantized_model_ir = ov.convert_model(quantized_model, example_input=dummy_input, input=[-1,3,input_width,input_height])

        ov.save_model(quantized_model_ir, "./int8.xml")
        model = core.compile_model(quantized_model_ir, 'CPU')
        print("End Profiling")
    else:
        import openvino.properties.hint as hints
        config = {
            "INFERENCE_PRECISION_HINT": "f16"
        }
        
        core.set_property(
            "CPU",
            {hints.execution_mode: hints.ExecutionMode.PERFORMANCE},
        )
        dummy_input = torch.randn(1, 3, input_width,input_height).float()
        model = core.compile_model(ov.convert_model(model, example_input=dummy_input),'CPU',config=config)
        


if export_onnx:
    #from onnxruntime.quantization import quantize_dynamic, QuantType
    from datetime import datetime 
    now = datetime.now()
    now = now.strftime("%Y-%m%d_%H-%M-%S")
    torch_input = torch.randn(1, 3, input_width, input_height)
    
    full_model_path = os.path.join("./", f"{now}_onnx_new_best_mbv2_ltrb_skp.onnx")
    #onnx_program = torch.onnx.dynamo_export(model, torch_input)
    #onnx_program.save("mbv2_shufflenet_widerface.onnx")
    torch.onnx.export(
    model.cpu(),
    torch_input,
    full_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11)
    # quantized_model_path = "224_onnx_new_best_mbv2_ltrb_skp_quantized.onnx"
    # quantize_dynamic(
    #     model_input="224_onnx_new_best_mbv2_ltrb_skp.onnx",           # Input ONNX model
    #     model_output=quantized_model_path, # Output quantized model
    #     weight_type=QuantType.QInt8        # Quantize weights to INT8
    # )
    # print(f"Quantized model saved to {quantized_model_path}")

if not cpu:
    model.cuda()
    
save_xml = False
if save_xml :
    if not os.path.isdir(os.path.join(folder,"annos")):
        os.mkdir(os.path.join(folder,"annos"))
    
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    class Converter():
        def __init__(self,save_folder):
            self.save_folder = save_folder
        def __call__(self,path,image_size,bboxes):
            annotation = self.create_pascal_voc_xml(path,image_size,bboxes)
            with open(self.save_folder+"/"+path.split(".")[0]+".xml","w") as f:
                f.write(annotation)
        def create_pascal_voc_xml(self,image_name, image_size, bboxes):
            """
            Creates Pascal VOC XML annotation for an image.
            
            Parameters:
                image_name (str): Name of the image file (e.g., 'image1.jpg').
                image_size (tuple): (width, height, depth) of the image.
                bboxes (list): List of bounding boxes in the format [[xmin, ymin, xmax, ymax, class]].
            
            Returns:
                str: XML string in Pascal VOC format.
            """
            
            # Initialize the XML structure
            annotation = ET.Element("annotation")
            
            # Folder (optional, can be skipped or customized as needed)
            folder = ET.SubElement(annotation, "folder")
            folder.text = "images"

            # File name
            filename = ET.SubElement(annotation, "filename")
            filename.text = image_name

            # Size (image dimensions)
            size = ET.SubElement(annotation, "size")
            width = ET.SubElement(size, "width")
            width.text = str(image_size[0])
            height = ET.SubElement(size, "height")
            height.text = str(image_size[1])
            depth = ET.SubElement(size, "depth")
            depth.text = str(image_size[2])

            # Objects (bounding boxes)
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, obj_class = bbox
                
                obj = ET.SubElement(annotation, "object")
                name = ET.SubElement(obj, "name")
                name.text = obj_class
                
                pose = ET.SubElement(obj, "pose")
                pose.text = "Unspecified"
                
                truncated = ET.SubElement(obj, "truncated")
                truncated.text = "0"
                
                difficult = ET.SubElement(obj, "difficult")
                difficult.text = "0"
                
                bndbox = ET.SubElement(obj, "bndbox")
                xmin_elem = ET.SubElement(bndbox, "xmin")
                xmin_elem.text = str(xmin)
                ymin_elem = ET.SubElement(bndbox, "ymin")
                ymin_elem.text = str(ymin)
                xmax_elem = ET.SubElement(bndbox, "xmax")
                xmax_elem.text = str(xmax)
                ymax_elem = ET.SubElement(bndbox, "ymax")
                ymax_elem.text = str(ymax)
            
            # Prettify XML
            xml_str = ET.tostring(annotation, encoding="utf-8")
            parsed_xml = minidom.parseString(xml_str)
            return parsed_xml.toprettyxml(indent="  ")
    convert = Converter(os.path.join(folder,"annos"))
    


if video:
    cap = cv2.VideoCapture(video_path)
    avg_fps = 0
    while 1:
        ret,img = cap.read()
        #img = cv2.resize(img,(1280,720))
        img = img[...,::-1]
        image,annos = infer_image(model,img,classes,stride,conf,half,input_shape=(input_height,input_width),cpu=cpu,openvino_exp=openvino_exp)
        #image = cv2.resize(image,(1280,720))
        #print(annos)
        cv2.imshow("ciou_iou_aware_centernet",image[...,::-1])
        ch = cv2.waitKey(1)
        if ch == ord("q"): break


else:
    files = glob(folder+"/*.jpg") + glob(folder+"/*.png") + glob(folder+"/*.JPG")
    for i in files:
        print(i)
        if save_xml:
            annotations = []
        image,annos = infer_image(model,i,classes,stride,conf,half,input_shape=(input_height,input_width),cpu=cpu,openvino_exp=openvino_exp)
        if save_xml:
            for b in annos:
                xmin = b[0]
                ymin = b[1]
                xmax = b[2]
                ymax = b[3]
                class_ = b[4]
                annotations.append([xmin,ymin,xmax,ymax,class_])
        if image.shape[1] > 1280: 
            image = cv2.resize(image,(1280,720))
        cv2.imshow("ciou_iou_aware_centernet",image[...,::-1])
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
        if ch == ord("s"): 
            if save_xml:
                size_ = cv2.imread(i).shape
                convert(i.split("\\")[-1],size_,annotations)
            else:
                continue
