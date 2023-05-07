import streamlit as st
from numpy import floor

from detect_picture_more import yolov5_use
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np
import os

def download(upload_file):
    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 展示图片
        st.image(opencv_image, channels="BGR")
        # 保存图片
        save_path=r"/home/luofan/kivy-venv/yolov5/mid"
        path=os.path.join( save_path,upload_file.name)
        cv2.imwrite(path, opencv_image)
        return path

if __name__ == '__main__':
    weights='runs/train/exp/weights/last.pt'  # model path or triton URL
    source='mid'  # file/dir/URL/glob/screen/0(webcam)
    imgsz=(640, 640)
    save_txt=False
    nosave=True
    classall=['0 ppb','10 ppb','100 ppb','10e3 ppb','10e4 ppb','10e5 ppb']
    st.markdown('层析分析(Machine Learning)')
    upload_file=st.file_uploader("请选择一张图片",help="上传应该为jpg或png格式图片",type=['png', 'jpg'],key="上传总量小于200M")
    mode=1
    if(mode==1):
        if upload_file is not None:
           if upload_file.name.endswith('.jpg') or upload_file.name.endswith('.png'):
               path = download(upload_file)
               if(st.button("OK")):
                   im0,result,result_soft=yolov5_use(weights,source,imgsz,save_txt,nosave)
                   if(result!=0):
                       st.image(im0, channels="BGR")
                       os.remove(path)
                       for i in range(0,len(result)):
                          st.success("分析完成")
                          st.write("预测结果是：",classall[result[i]])
                          st.write("预测概率是：",str(floor(result_soft[i].detach().numpy()*100)[0]),'%')
                   else:
                       st.warning("Warning")
                       st.write("没有目标!(noting:可能为拍照不清晰导致)")
                       os.remove(path)
                   
           else:
               st.write("选中不是图片，请重新选择！")


# streamlit run appuse.py




