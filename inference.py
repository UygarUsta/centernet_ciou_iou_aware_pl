import torch
import numpy as np
import cv2
from infer_utils import infer_image,load_model,hardnet_load_model
from glob import glob
import os
import time 
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

video = False
half = True 
cpu = False 
trace = False 
openvino_exp = False 
openvino_int8 = False 
export_onnx = False 
save_annotations = True
nms_ops = True

f = open("classes_ekar.txt","r").readlines()
classes = []
for i in f:
    classes.append(i.strip('\n'))

print(classes)

input_height = 768
input_width = 768
stride = 4
folder =  r"C:\Users\uygar.usta\Desktop\evrak_detection\test\21\veraset_ilanı"  #r"C:\Users\uygar.usta\Desktop\codes\focal_instance_segmentation\valid" 
video_path = 0
model_path = r"ekar_768_best_model_mAP_0.8859.pth"
device = "cuda"
model_type = "hardnet"

if model_type == "mbv4_timm":
    conf = 0.35
    from mbv4_timm import CenterNet
    model = CenterNet(nc=len(classes))
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "hardnet":
    conf = 0.35
    from hardnet import get_pose_net
    model = get_pose_net(68,{"hm":len(classes),"wh":2,"offset":2,"iou":1}) 
    if model_path.endswith(".ckpt"):
        #model = hardnet_load_model(model,model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") 
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(model_path))
    
        



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
    

if save_annotations :
    import json 
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    class Converter():
        def __init__(self, save_folder):
            self.save_folder = save_folder
            # Ensure the directory exists before writing
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

        def __call__(self, path, image_size, bboxes, format="json"):
            """
            Processes and writes the file.
            format: "json" for LabelMe, "xml" for Pascal VOC
            """
            # Get filename without extension (e.g., 'cat.jpg' -> 'cat')
            file_id = os.path.splitext(os.path.basename(path))[0]
            
            if format.lower() == "json":
                self.save_json(file_id, path, image_size, bboxes)
            else:
                self.save_xml(file_id, path, image_size, bboxes)

        def save_json(self, file_id, image_name, image_size, bboxes):
            """Generates and writes LabelMe JSON file."""
            data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": [],
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": image_size[1],
                "imageWidth": image_size[0]
            }

            for bbox in bboxes:
                xmin, ymin, xmax, ymax, obj_class = bbox
                shape = {
                    "label": obj_class,
                    "points": [[float(xmin), float(ymin)], [float(xmax), float(ymax)]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                data["shapes"].append(shape)

            file_path = os.path.join(self.save_folder, f"{file_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Successfully saved: {file_path}")

        def save_xml(self, file_id, image_name, image_size, bboxes):
            """Generates and writes Pascal VOC XML file."""
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "filename").text = image_name
            
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(image_size[0])
            ET.SubElement(size, "height").text = str(image_size[1])
            ET.SubElement(size, "depth").text = str(image_size[2])

            for bbox in bboxes:
                xmin, ymin, xmax, ymax, obj_class = bbox
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = obj_class
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

            # Prettify and write
            xml_str = ET.tostring(annotation, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            
            file_path = os.path.join(self.save_folder, f"{file_id}.xml")
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(pretty_xml)
            print(f"Successfully saved: {file_path}")
    convert = Converter(folder)
    


if video:
    cap = cv2.VideoCapture(video_path)
    avg_fps = 0
    while 1:
        ret,img = cap.read()
        #img = cv2.resize(img,(1280,720))
        img = img[...,::-1]
        img = Image.fromarray(img)
        image,annos = infer_image(model,img,classes,stride,conf,half,input_shape=(input_height,input_width),cpu=cpu,openvino_exp=openvino_exp,nms=nms_ops)
        #image = cv2.resize(image,(1280,720))
        #print(annos)
        cv2.imshow("ciou_iou_aware_centernet",image[...,::-1])
        ch = cv2.waitKey(1)
        if ch == ord("q"): break


else:
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = glob(os.path.join(folder, "*"))
    file_len = len(files)
    for ind, file_path in enumerate(files):
        # Splitext returns (root, ext) -> e.g., ('/path/to/image', '.JPG')
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in VALID_EXTENSIONS:
            continue
        
        print(f"[{ind}/{file_len}]")
        print(f"Processing: {file_path}")
        if save_annotations:
            annotations = []
        image,annos = infer_image(model,file_path,classes,stride,conf,half,input_shape=(input_height,input_width),cpu=cpu,openvino_exp=openvino_exp,nms=nms_ops)
        if save_annotations:
            for b in annos:
                xmin = b[0]
                ymin = b[1]
                xmax = b[2]
                ymax = b[3]
                class_score = b[4]
                class_ = class_score.split(" ")[0]
                annotations.append([xmin,ymin,xmax,ymax,class_])
        if image.shape[1] > 1280: 
            image = cv2.resize(image,(1280,720))
        cv2.imshow("ciou_iou_aware_centernet",image[...,::-1])
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
        if ch == ord("s"): 
            if save_annotations:
                size_ = cv2.imread(file_path).shape
                convert(file_path.split("\\")[-1],size_,annotations,'json')
            else:
                continue
