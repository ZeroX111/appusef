import argparse
import os
import platform
import sys
from pathlib import Path
from torchvision import transforms
from PIL import Image
from numpy import floor
import streamlit as st

import torch
from tranendmodel import Mytrain

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


@smart_inference_mode()
def yolov5_use2(
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
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
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
       save_dir_txt='/home/luofan/kivy-venv/yolov5/mid3/mid.txt'
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
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 后端流程
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    num=0
    bar=st.progress(num/numpic*100)
    im0_all=[]
    result_f_re_all=[]
    result_soft_re_all=[]
   
    for path, im, im0s, vid_cap, s in dataset:   #图片具体路径，图片数据(后续被缩小+正则化）,原图片数据,是否为视频,图片路径      dataset过程中就读取了所有图片
        #图片预处理
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 代入模型预测结果
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
                p, im0, frame = path[i], im0s[i].copy(), dataset.count   #图片帧
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            #存储路径
            save_path = str(save_dir / p.name)  # im.jpg
            
            save_dir_txt='/home/luofan/kivy-venv/yolov5/mid3'
            
            #txt_path = str(save_dir_txt / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
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
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string   names种类集合          s时间

                # Write results  
                result_f_re=[]
                result_soft_re=[]
                for *xyxy, conf, cls in reversed(det):   #修改后框位置 ，置信度，种类 
                    # crop图片并进行分类
                    paths=save_one_box2(xyxy, imc, path_mid_t,BGR=True,save=True)  # crop图片并保存函数    ？：？：3格式
                    #print(paths)
                    img_tensor=crop_to_tensor(paths)              
                    mytrain=Mytrain()
                    mytrain=torch.load(pth_path)  
                    result_pr=mytrain(img_tensor)
                    result_f=result_pr.argmax(1)    #最终分类
                      
                    #概率
                    softmax_0= torch.nn.Softmax(dim=0)
                    result_soft=softmax_0(result_pr[0])       #输出维度一般都扩1注意  忘记
                    
                    result_f_re.append(result_f)
                    result_soft_re.append(result_soft[result_f])
                    #是否保存结果为txt
                    if save_txt:  # Write to file
                        cls=classfinal[result_f]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        txt_path ='/home/luofan/kivy-venv/yolov5/mid3/mid'
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write('%s ' % p.name)
                            f.write(('%s ' * len(line)).rstrip() % line + '\n')
                    #是否保存处理后图片
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(result_f)  # integer class 
                        label = None if hide_labels else (classfinal[c] if hide_conf else f'{classfinal[c]} {conf:.2f}')  #种类名
                        annotator.box_label(xyxy, label, color=colors(c, True))   # 标记图片并保存函数
                          
            #标记后图片                noting  2  !!!!
            im0 = annotator.result()   
            
            
            if len(det)==0:  
                im0=0
                result_f_re=0
                result_soft_re=0
            
            im0_all.append(im0)
            result_f_re_all.append(result_f_re)
            result_soft_re_all.append(result_soft_re)
            
            # Save results (image with detections)  是否保存图片
            if save_img:
                cv2.imwrite(save_path, im0)   #直接保存
                
            num=num+1
            bar.progress(int(num/numpic*100))
            
    # Print time (inference-only)
    #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    #t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
            
    #print(len(det))
    return im0_all,result_f_re_all,result_soft_re_all
        

if __name__ == '__main__':
    weights=ROOT / 'runs/train/exp/weights/last.pt'  # model path or triton URL
    source='/home/luofan/kivy-venv/yolov5/mid2/test'  # file/dir/URL/glob/screen/0(webcam)
    #save_crop=True  # save cropped prediction boxes
    imgsz=(640, 640)
    save_txt=True
    nosave=True
    im0,classfinal,result_soft=yolov5_use2(weights,source,imgsz,save_txt,nosave)
    print(im0[0].shape)
    #print(len(classfinal))
    #print(floor(result_soft[0].detach().numpy()*100)[0])
