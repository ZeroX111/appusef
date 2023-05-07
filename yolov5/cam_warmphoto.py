'''
CAM方法分析
'''
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
import cv2
import numpy as np
import os

def draw_CAM(mytrain, img_path, path_warm, name,target_layer):
    # # imread返回从指定路径加载的图像
    # rgb_img1 = cv2.imread(image_path[0], 1)  # imread()读取的是BGR格式   RBG
    img = Image.open(img_path[0])  ##绝对地址
    tran_re = transforms.Resize((50, 50))
    img_resized = tran_re(img)
    rgb_img1 = img_resized.convert('RGB')  # 将png转为jpg,删除透明度通道
    # rgb_img1.show()
    rgb_img = np.float32(rgb_img1) / 255

    # preprocess_image作用：归一化图像，并转成tensor        可能需要修改  --------------------------------------------------------------------------------------------------------
    input_tensor = preprocess_image(rgb_img, mean=[0.406, 0.456, 0.485],
                                    std=[0.225, 0.224, 0.229])  # torch.Size([1, 3, 224, 224])

    '''
    3)初始化CAM对象，包括模型，目标层以及是否使用cuda等
    '''
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=mytrain, target_layers=target_layer, use_cuda=False)

    '''
    4)选定目标类别，如果不设置，则默认为分数最高的那一类
    '''
    target_category = None

    '''
    5)计算cam
    '''
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)  # [batch, 224,224]

    # ----------------------------------
    '''
    6)展示热力图并保存
    '''
    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam,image_weight=0.5,use_rgb = True)  # (224, 224, 3)

    #提取图片
    img_path_new = os.path.join(path_warm, name[0])
    # visualization =visualization.numpy()
    visualization=Image.fromarray(visualization)  #实现array到image转化 numpy to  PIL
    visualization.save(img_path_new)
    visualization.save(img_path_new,qualiity=95)
    # cv2.imwrite(img_path_new, visualization)

    # tensorboard展示
    visualization_tran2 = transforms.ToTensor()
    visualization_tensor = visualization_tran2(visualization)
    photo = visualization_tensor.reshape([-1, 3, 50, 50])

    return photo #便于tensorboard接收