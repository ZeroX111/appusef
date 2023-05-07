from torch.utils.tensorboard import SummaryWriter
from code.tran3model import *
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import numpy as np

#使用GPU训练
if torch.cuda.is_available(): print("gpu可以用")
else: print("gpu不可以用")
#定义训练设备
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #显卡用：cuda:0

#加载数据并明清长度---自定义数据集合      96
class  Mytrain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #Sequential简化代码
        self.model1=Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
            MaxPool2d(2, stride=2, padding=0),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2, stride=2, padding=0),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

        # self.Linear1 = Linear(196608,10) #in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0

    def  forward(self, x):
        x=self.model1(x)
        return x


dataset_test=torchvision.datasets.CIFAR10(root=r"D:\pytorch_train\deeplearning\datasetdownload",train=False,transform=torchvision.transforms.ToTensor(),download=True)  #复制  train=False

length_test=len(dataset_test)
print("测试数据集的长度：{}".format(length_test) )

#dataloader打包
dataloader_test=DataLoader(dataset_test, batch_size=1,shuffle=False,drop_last=True)   #展平需要去掉最后的以便同维

#引入已搭建的模型
mytrain=Mytrain()

#查看层名
for name in mytrain.named_modules():
    print(name)

#读取数据
mytrain=torch.load(r"D:\pytorch_train\deeplearning\rabish_test\mytrain_99.pth")#-------------------------------------------------------------------------------------------

#添加tensorboard
wit=SummaryWriter("logs_test_photo_warm")

#步数
step=1;

for data in dataloader_test:
    imgs, targets= data  # 注意：提取图片输入
    imgs = imgs.to(device)  # 利用GPU计算数据
    targets = targets.to(device)
    output = mytrain(imgs)
    print(step)
    print("图像原种类：{}".format(targets))
    print("预测种类：{}".format(output.argmax(1)))

    # 输出热力图像
    # 2.选择目标层:最后一个卷积层
    target_layer = [mytrain.model1[4]]#----------------------------------------------------------------------------------------------------------------------------------

    # # imread返回从指定路径加载的图像
    # rgb_img = imgs.reshape([32, 32,3])
    # img_tran = transforms.ToPILImage()
    # rgb_img1 = img_tran(rgb_img)
    # rgb_img1.show()

    rgb_img2 = imgs.reshape([3, 50, 50])
    img_tran = transforms.ToPILImage()
    rgb_img1 = img_tran(rgb_img2)
    rgb_img1 = rgb_img1.convert('RGB')
    # rgb_img1.show()
    # img_tran2 = transforms.ToTensor()
    # rgb_img3 = img_tran2(rgb_img1)
    # rgb_img3=rgb_img3.reshape([-1,3,32,32])
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
    visualization = show_cam_on_image(rgb_img, grayscale_cam,image_weight=0.7)  # (224, 224, 3)

    #提取图片
    path_warm = rf'D:\pytorch_train\deeplearning\photo_warm_test'
    img_path_new = os.path.join(path_warm, name[0])
    # visualization =visualization.numpy()
    visualization=Image.fromarray(visualization)
    visualization.save(img_path_new,qualiity=95)
    # cv2.imwrite(img_path_new, visualization)

    #tensorboard展示
    visualization_tran2 = transforms.ToTensor()
    visualization_tensor = visualization_tran2(visualization)
    photo = visualization_tensor.reshape([-1,3,32,32])


    rgb_tran=transforms.ToTensor()
    rgb_img1=rgb_tran(rgb_img1)
    rgb_img1 = rgb_img1.reshape([-1, 3, 32, 32])

    wit.add_images('test_or', rgb_img1, step)  # name[0]

    wit.add_images('test', photo, step)#name[0]
    step=step+1
    if step>10:
        break

wit.close()


# tensorboard --logdir=logs_test_photo_warm --port=6007