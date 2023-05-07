import streamlit as st
from numpy import floor

from detect_picture_more import yolov5_use
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
from detect_picture_more2 import yolov5_use2
from detect_picture_more3 import yolov5_use3
import torch
import numpy as np
import os
import zipfile
import av
import shutil
from streamlit_webrtc import webrtc_streamer,VideoTransformerBase
from scipy import misc

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box,save_one_box2
from utils.torch_utils import select_device, smart_inference_mode


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

def download(upload_file,save_path=r"/home/luofan/kivy-venv/yolov5/mid"):
    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 保存图片
        path=os.path.join( save_path,upload_file.name)
        st.session_state.name.append(upload_file.name)
        cv2.imwrite(path, opencv_image)
        return path,opencv_image
        
def download_zip(upload_file,save_path=r"/home/luofan/kivy-venv/yolov5/mid2"):
    if upload_file is not None:
        file_name=upload_file.name.split(".")[0]
        path=os.path.join(save_path,file_name)
        with zipfile.ZipFile(upload_file,"r") as z:
           z.extractall(path=path)
        return path
        
def alanye(weights,imgsz,save_txt,nosave,picshow,number,path='.',picture_num1=1):
    if(st.button("OK")):
         #st.progress(10)   需要参数
         im0,result,result_soft=yolov5_use2(weights,path,imgsz,save_txt,nosave)
         st.session_state.im0=im0
         st.session_state.result=result
         st.session_state.result_soft=result_soft
         st.session_state.files=os.listdir(path)
         st.session_state.numpic=len(st.session_state.files)
         st.session_state.state.append(1)
     
     
    if(st.session_state.state!=[]):
         if(st.session_state.numpic>1):
               number=st.sidebar.slider('选择第几张照片',1,st.session_state.numpic) 
         st.success("分析完成")   
         if(st.session_state.result[number-1]!=0):
          #shutil.rmtree(path)
               if(picshow=='是'):st.image(st.session_state.im0[number-1], channels="BGR")
               for i in range(0,len(st.session_state.result[number-1])):
                   if(i==0 and mode=="照片分析"):st.write("图片：",st.session_state.name[number-1])
                   st.write("预测结果是：",classall[st.session_state.result[number-1][i]])
                   st.write("预测概率是：",str(floor(st.session_state.result_soft[number-1][i].detach().numpy()*100)[0]),'%')
         else:
               st.warning("Warning") 
               st.write("没有目标!(noting:可能为拍照不清晰导致)") 



class VideoProcessor:
    def recv(self,frame):
         img=frame.to_ndarray(format="bgr24")
        #  source=r"/home/luofan/kivy-venv/yolov5/mid4"  # file/dir/URL/glob/screen/0(webcam)

        #  dataset=LoadImages2(img, source,img_size=(640, 640), stride=self.stride, auto=self.pt, vid_stride=1)
        #  seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

         #图片预处理
        #  print(4)
        #  im, im0s = dataset[0]
        #  print(3)
        #  with dt[0]:
        #  im = torch.from_numpy(img).to(model.device)
        #  print(2)
        #  img = img.half() #if model.fp16 else im.float()  # uint8 to fp16/32
        #  print(img)
         img /= 255  # 0 - 255 to 0.0 - 1.0
         print(2)
         img = img[None]  # expand for batch dim

         print(img.shape)
        #  print(self.model)

         # 代入模型预测结果
         pred = st.session_state.model(img, augment=False, visualize=False)
         print(1)


         # NMS处理得最终结果
         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
         im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh    宽长 4维
         annotator = Annotator(im0, line_width=line_thickness, example=str(names))  #准备画框         
         #标记后图片                noting  2  !!!!
         im0 = annotator.result()

         return av.VideoFrame.from_ndarray(im0,format="bgr24")#im0[0]
      
               

if __name__ == '__main__':
    weights='runs/train/exp/weights/best.pt'  # model path or triton URL
    source='mid'  # file/dir/URL/glob/screen/0(webcam)
    imgsz=(640, 640)
    save_txt=False
    nosave=True
    classall=['0 CFU/mL','10 CFU/mL','100 CFU/mL','10e3 CFU/mL','10e4 CFU/mL','10e5 CFU/mL']
    st.sidebar.markdown('层析分析(Machine Learning)')
    st.balloons()
    mode=st.sidebar.selectbox("选择模式",["照片分析","数据包分析","实时分析"])
    st.session_state.state=[]
    print(torch.cuda.is_available())
    
    if(mode=="照片分析"):
        upload_file=st.file_uploader("请选择一张图片",help="上传应该为jpg或png格式图片",type=['png', 'jpg'],key="上传总量小于200M",accept_multiple_files=True) #
        picshow=st.sidebar.radio("是否查看处理好的图片",['是','否'])
        st.session_state.path_all=[]
        st.session_state.opencv_image=[]
        st.session_state.name=[]
        picture_num=len(upload_file)
        #print(picture_num)
        flag=0
        if upload_file is not None:
           for i in range(0,picture_num):
               if (upload_file[i].name.endswith('.jpg') or upload_file.name[i].endswith('.png'))!=1:
                   flag=1
               else:
                   path,opencv_image= download(upload_file[i])
                   st.session_state.path_all.append(path)
                 
           if flag==0:
                   #st.session_state.opencv_image.append(opencv_image)
               save_path=r"/home/luofan/kivy-venv/yolov5/mid"
               number=1
               alanye(weights,imgsz,save_txt,nosave,picshow,number,path=save_path,picture_num1=picture_num)                   
               if(len(st.session_state)>4):
                   for i in range(0,picture_num):
                       os.remove(st.session_state.path_all[i]) 
           else:
               st.write("选中不是图片，请重新选择！")
               
    elif(mode=="数据包分析"):
        #上传数据包则返回txt文件
        #lookway=st.sidebar.radio("选择查看结果的方式",['数值','输入'])
        upload_file=st.file_uploader("请上传zip数据压缩包",help="上传应该为zip格式包",type=['zip'],key="上传总量小于200M")
        picshow=st.sidebar.radio("是否查看处理好的图片",['是','否'])
        txtdown=st.sidebar.radio("是否输出txt文件",['是','否'])
        if upload_file is not None:
           if upload_file.name.endswith('.zip'):# or upload_file.name.endswith('.png'):
               path = download_zip(upload_file)
               if(txtdown=='是'):
                  save_txt=True
               else:
                  save_txt=False
               number=1
               alanye(weights,imgsz,save_txt,nosave,picshow,number,path=path)   
               if(txtdown=='是' and len(st.session_state)>4):
                  with open('/home/luofan/kivy-venv/yolov5/mid3/mid.txt') as file:
                     st.download_button(label="点击下载",data=file,file_name="分析结果.txt")
               shutil.rmtree(path)
           else:
               st.write("选中不是zip文件，请重新选择！")
               
    elif(mode=="实时分析"):
         st.write("实时监测分析")

         weights='runs/train/exp/weights/best.pt'  # model path or triton URL
         source=r"/home/luofan/kivy-venv/yolov5/mid4"  # file/dir/URL/glob/screen/0(webcam)
         imgsz=(640, 640)
         conf_thres=0.25,  # confidence threshold
         iou_thres=0.45,  # NMS IOU threshold
         max_det=1000,  # maximum detections per image
         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
         save_conf=False,  # save confidences in --save-txt labels
         classes=None,  # filter by class: --class 0, or --class 0 2 3
         agnostic_nms=False,  # class-agnostic NMS
         augment=False,  # augmented inference
         visualize=False,  # visualize features
         update=False,  # update all models
         name='exp',  # save results to project/name
         exist_ok=True,  # existing project/name ok, do not increment
         line_thickness=3,  # bounding box thickness (pixels)
         hide_labels=False,  # hide labels
         hide_conf=False,  # hide confidences
         half=False,  # use FP16 half-precision inference
         dnn=False,  # use OpenCV DNN for ONNX inference
         vid_stride=1,  # video frame-rate stride
         data='yolov5/data/coco128.yaml',  # dataset.yaml path
         device=''
         device = select_device(device)
         model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
         st.session_state.stride, st.session_state.names,st.session_state.pt = model.stride, model.names, model.pt
        #imgsz = check_img_size(imgsz, s=stride)  # check image size

        # 数据打包
         bs = 1  # batch_size
         #dataset=LoadImages2(img, source,img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        
        # 后端流程
         model.warmup(imgsz=(1 if st.session_state.pt or model.triton else bs, 3, *imgsz))  # warmup
         st.session_state.model=model
         st.session_state.classfinal=['0 ppd','10 ppd','100 ppd','10e3 ppd','10e4 ppd','10e5 ppd']     #定义最终种类



         webrtc_streamer(key="example",video_processor_factory=VideoProcessor,async_transform=True)
        

# python3 -m streamlit run appuse.py




