import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
###########################基本的RNN架构###########################
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.input_to_hidden=nn.Linear(input_size,hidden_size)
        self.hidden_to_hidden=nn.Linear(hidden_size,hidden_size)
        self.hidden_to_output=nn.Linear(hidden_size,output_size)
        #定义不同层之间的线性变换
    def forward(self,x,hidden):
        for t in range(x.size(1)):
            input_t=x[:,t,:]
            hidden=torch.relu(self.input_to_hidden(input_t)+self.hidden_to_hidden(hidden))#对比relu tanh sigmoid激活函数 relu和tanh准确率效果接近 sigmoid效果较差 relu所用时间略少于tanh
            output=self.hidden_to_output(hidden)

        return output,hidden
###################################################################
    
    
def train(model,dataset,loss,optimizer,hidden_size=128,times=10):
    train_losses=[]
    for i in range(times):
        total_loss=0
        for images,labels in dataset:
            images=images.view(-1,28,28)#图片形状为(128,1,28,28) 由于数据集中图为灰度图 仅有一个颜色通道 可调整其形状
            
            hidden=torch.zeros(images.size(0),hidden_size)
            
            output,hidden=model(images,hidden)
            
            los=loss(output,labels)
            
            optimizer.zero_grad()
            los.backward()#反向传播
            optimizer.step()#更新模型参数
            total_loss+=los.item()
        
        avg_loss=total_loss/len(dataset)
        train_losses.append(avg_loss)
        print(f'第{i+1}/{times}轮，平均损失为{avg_loss:.4f}')
    return train_losses
            
            
def loss_show(loss):
    x=list(range(1, len(loss) + 1))
    plt.plot(x,loss)
    plt.title("Training Loss")
    plt.xlabel("Time")
    plt.ylabel("Average Loss")
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.show()
#可视化训练过程

def test(model,dataset,hidden_size=128):
    correct=0
    total=0
    for images,labels in dataset:
        images=images.view(-1,28,28)
        hidden=torch.zeros(images.size(0),hidden_size)
        output,hidden=model(images,hidden)
        _,predictions=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(labels==predictions).sum()
    accuracy=correct/total*100
    print(f'准确率为{accuracy:.2f}%')
    return accuracy #可视化结果
   

input_size=28
hidden_size=256 
output_size=10  

model=RNN(input_size, hidden_size, output_size)
loss=nn.CrossEntropyLoss()
#以交叉熵损失作为指标计算函数
optimizer=optim.Adam(model.parameters(), lr=0.0002)
#采用Adam优化器以动态调整学习率
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
#对图像数据进行归一化 标准化
train_dataset=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)

train_loader=DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)#对于训练过程每次打乱顺序以提高模型稳定性
test_loader=DataLoader(dataset=test_dataset,batch_size=128,shuffle=False)

start_time = time.time()
train_losses=train(model,train_loader,loss,optimizer,hidden_size=hidden_size,times=10)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"训练用时:{elapsed_time:.2f}秒")

test_accuracy=test(model,test_loader,hidden_size=hidden_size)

loss_show(train_losses)
