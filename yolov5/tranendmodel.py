import  torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Flatten,MaxPool2d,Conv2d,Linear,Sequential,BatchNorm2d,Softmax,ReLU,Dropout,AvgPool2d
import pandas as pd
import numpy as np

#Dropout(p=0.5,inplace=False)  #不节省存储




#res网络则是在forward中用x=x+block()形式  block自定义函数   or 老实连接即可    ----   forward 都可以支持   block内为cov和BN 残差后再激活，之前也放激活

class  Mytrain(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1:Conv2d=Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)  # 本来减少为kenel宽-1
        # self.maxpool1=MaxPool2d(2, stride=2, padding=0)
        # self.conv2=Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.max2=MaxPool2d(2, stride=2, padding=0)
        # self.conv3=Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # self.max3=MaxPool2d(2, stride=2, padding=0) # 一般减半操作，stride=kernel边长
        # self.flatten=Flatten()
        # self.linear1=Linear(2304, 64)
        # self.linear2=Linear(64, 7)
        # Sequential简化代码
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # 本来减少为kenel宽-1
            MaxPool2d(2, stride=2, padding=0),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2, stride=2, padding=0),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2, stride=2, padding=0),  # 一般减半操作，stride=kernel边长
            Flatten(),
            Linear(2304, 64),
            Linear(64, 7)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.max2(x)
        # x = self.conv3(x)
        # x = self.max3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x=self.model1(x)
        return x


## 定义导出函数
def parm_to_excel(excel_name,sheet,parm):
    with pd.ExcelWriter(excel_name,mode='a',if_sheet_exists='replace') as writer:
        if(sheet.endswith('weight') and len(parm.shape)>2):
            [output_num,input_num,filter_size,_]=parm.size()
            for i in range(output_num):
                for j in range(input_num):
                    data=pd.DataFrame(parm[i,j,:,:].detach().numpy())
                    #print(data)
                    data.to_excel(writer,sheet,header=False,index=False,startrow=j*(filter_size),startcol=i*(filter_size))
        elif (sheet.endswith('weight') and len(parm.shape) == 2):
            [num, bias] = parm.size()
            for i in range(num):
                data = pd.DataFrame(parm[i , :].detach().numpy())
                # print(data)
                data.to_excel(writer, sheet,header=False,index=False,startrow=0, startcol=i)


        elif(sheet.endswith('bias')):
            parm=parm.reshape([-1,parm.size(0)])
            [num,bias] = parm.size()
            for i in range(num):
                    data = pd.DataFrame(parm[i,:].detach().numpy())
                    #print(data)
                    data.to_excel(writer, sheet,header=False,index=False, startrow=0, startcol=i)
        else:
            print("run")
        writer.save()


if __name__ == '__main__':
    mytrain=Mytrain()
    # input=torch.ones((64,3,32,32))
    input = torch.ones((1, 3, 50, 50))
    mytrain = torch.load(r"D:\pytorch_train\deeplearning\data_value\mytrainend2.pth")
    parm = {}
    num=0
    nameall=[]
    for name, parameters in mytrain.named_parameters():
        nameall.append(name)
        print(name, ':', parameters.size())
        print('------------------------')
        #parm[name] = parameters.detach().numpy()
        parm_to_excel('para.xlsx',name,parameters)
        num=num+1

    print(nameall)
    file_path=r"D:\pytorch_train\deeplearning\para.xlsx"
    raw_data=pd.read_excel(file_path,nameall[0])
    print(raw_data)


    output = mytrain(input)
    # for _,param in enumerate(mytrain.named_parameters()):
    #     print(param)
    #     print('------------------------')


    print(output)
    print(output.shape)
    # output2=mytrain.weight_out()
    # print(output2.shape)

    # 作出结构图
    # wit=SummaryWriter("logs10")
    # wit.add_graph(kk,input)
    # wit.close()



# Sequential简化代码
        # self.model1=Sequential(
        #     Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
        #     MaxPool2d(2, stride=2, padding=0),
        #     Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     MaxPool2d(2, stride=2, padding=0),
        #     Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        #     MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
        #     Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        #     MaxPool2d(2, stride=2, padding=0),
        #     Flatten(),
        #     Linear(576,64),
        #     Linear(64, 7)
        # )

        # self.Linear1 = Linear(196608,10) #in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0


# x=self.model1(x)


#1、该网络正确率最高74%  输入100X100
# Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
# MaxPool2d(2, stride=2, padding=0),
# Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
# MaxPool2d(2, stride=2, padding=0),
# Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
# MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
# Flatten(),
# Linear(9216, 64),
# Linear(64, 7)

#2、该网络正确率最高82%   输入50X50    证明卷积大小与分辨率之间需要相互考虑猜测成立
# Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
# MaxPool2d(2, stride=2, padding=0),\
# Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),\
# MaxPool2d(2, stride=2, padding=0),
# Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
# MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
# Flatten(),
# Linear(,64),
# Linear(64, 7)

#3、该网络正确率最高80%   输入32X32    一般分辨率越高越好，理论上对于这种看具体数值的更是如此
# Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
# MaxPool2d(2, stride=2, padding=0),\
# Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),\
# MaxPool2d(2, stride=2, padding=0),
# Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
# MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
# Flatten(),
# Linear(1024,64),
# Linear(64, 7)


#4、该网络正确率最高76%   输入32X32    一般分辨率越高越好，理论上对于这种看具体数值的更是如此
# Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),    #本来减少为kenel宽-1
# MaxPool2d(2, stride=2, padding=0),\
# Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),\
# MaxPool2d(2, stride=2, padding=0),
# Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
# MaxPool2d(2, stride=2, padding=0),     #一般减半操作，stride=kernel边长
# Flatten(),
# Linear(1024,64),
# Linear(64, 7)
