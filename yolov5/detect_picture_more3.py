import argparse
import os
import platform
import sys
from pathlib import Path
from torchvision import transforms
from PIL import Image
from numpy import floor
import streamlit as st
import numpy as np

import torch
from tranendmodel import Mytrain
from utils.augmentations import letterbox
                                 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box,save_one_box2
from utils.torch_utils import select_device, smart_inference_mode

def crop_to_tensor(path):
       img = Image.open(path)  ##绝对地址
       os.remove(path)
       tran_re = transforms.Resize((50, 50))
       img_resized = tran_re(img)
       img_resized = img_resized.convert('RGB')  # 将png转为jpg,删除透明度通道
       img_tran2 = transforms.ToTensor()
       img_tensor = img_tran2(img_resized)
       img_tensor = torch.unsqueeze(img_tensor,0)
       return img_tensor


class LoadImages2:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, img, path,img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        self.img=img
        self.img_size = img_size
        self.stride = stride
        self.nf = 1
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        # Read image
        self.count += 1
        im0 = self.img  # BGR

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return im, im0   

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


@smart_inference_mode()
def yolov5_use3(
        img,
        weights,  #=ROOT / 'runs/train/exp/weights/last.pt',  # model path or triton URL
        source,  #=ROOT / 'inference',  # file/dir/URL/glob/screen/0(webcam)
        imgsz,#=(640, 640),  # inference size (height, width)
        save_txt,   #=False,  # save results to *.txt
        nosave,  #=False,  # do not save images/videos
        path_mid_t=ROOT / 'runs/midcrop/mid',   #维度转化中间数据
        pth_path=ROOT / 'data/mytrainend2.pth', #pth存储路径
        save_crop=True,  # save cropped prediction boxes
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='1',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #读取路径
    source = str(source)
    files=os.listdir(source)
    numpic=len(files)
    
    #清空txt文件内容
    if save_txt:
       save_dir_txt='yolov5/mid3/mid.txt'
       file=open(save_dir_txt,'w').close()

    #判断是否为网页图片，或者屏幕（实时检测）    是否保存       注意screen属性
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  #判断是否为数据流临时文件等文件
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # 保存路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 读取模型 检查图片是否符合要求
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # 数据打包
    bs = 1  # batch_size
    dataset=LoadImages2(img, source,img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    

    # 后端流程
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    num=0
    bar=st.progress(num/numpic*100)
    im0_all=[]
    result_f_re_all=[]
    result_soft_re_all=[]
   
    for im, im0s in dataset:   #图片具体路径，图片数据(后续被缩小+正则化）,原图片数据,是否为视频,图片路径      dataset过程中就读取了所有图片
        #图片预处理
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 代入模型预测结果
        path='/home/luofan/kivy-venv/yolov5/mid4/out.jpg'
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS处理得最终结果
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        classfinal=['0 ppd','10 ppd','100 ppd','10e3 ppd','10e4 ppd','10e5 ppd']     #定义最终种类
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # 预测结果处理
        
        for i, det in enumerate(pred):  # per image   图片下标    boxs坐标+置信度+种类
            seen += 1
            if webcam:  # batch_size >= 1
                im0, frame = im0s[i].copy(), dataset.count   #图片帧
            else:
                im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
            #存储路径
            save_dir_txt='/home/luofan/kivy-venv/yolov5/mid3'
            
            #print('s='.format(s))
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh    宽长 4维
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  #准备画框
            if len(det):
                # Rescale boxes from img_size to im0 size           box大小映射
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()#取变换后长宽01（四维变2维） box数据  原始图片形状 

                # Print results  目标数量 
                for c in det[:, 5].unique():  #由大到小排列
                    n = (det[:, 5] == c).sum()  # detections per class  目标数

                # Write results  
                result_f_re=[]
                result_soft_re=[]
                for *xyxy, conf, cls in reversed(det):   #修改后框位置 ，置信度，种类 
                    # crop图片并进行分类
                    pass
                    #paths=save_one_box2(xyxy, imc, path_mid_t,BGR=True,save=True)  # crop图片并保存函数    ？：？：3格式
                    #print(paths)
                    #img_tensor=crop_to_tensor(paths)              
                    #mytrain=Mytrain()
                    #mytrain=torch.load(pth_path)  
                    #result_pr=mytrain(img_tensor)
                    #result_f=result_pr.argmax(1)    #最终分类
                      
                    #概率
                    #softmax_0= torch.nn.Softmax(dim=0)
                    #result_soft=softmax_0(result_pr[0])       #输出维度一般都扩1注意  忘记
                    
                    #result_f_re.append(result_f)
                    #result_soft_re.append(result_soft[result_f])
                               
            #标记后图片                noting  2  !!!!
            im0 = annotator.result()   
            
            
            if len(det)==0:  
                im0=0
                result_f_re=0
                result_soft_re=0
                target=0
            else:
                target=1
            
            #im0_all.append(im0)
            #result_f_re_all.append(result_f_re)
            #result_soft_re_all.append(result_soft_re)
            
    return im0,target
        

if __name__ == '__main__':
    img=cv2.imread('yolov5/inference/22(1).jpg')
    weights=ROOT / 'runs/train/exp/weights/last.pt'  # model path or triton URL
    source='yolov5/mid2/test'  # file/dir/URL/glob/screen/0(webcam)
    #save_crop=True  # save cropped prediction boxes
    imgsz=(640, 640)
    #save_txt=True
    #nosave=True
    #im0,tar=yolov5_use3(img,weights,source,imgsz,save_txt,nosave)
    #print(im0.shape)
    #print(len(classfinal))
    #print(floor(result_soft[0].detach().numpy()*100)[0])
