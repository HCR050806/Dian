import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# 保存原始图像与去噪图像
def save_image(image0,image,t,idx):
    image0=image0.cpu().detach().numpy()
    image=image.cpu().detach().numpy()
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image0)
    ax[0].set_title(f'Original Image{idx+1}')
    ax[0].axis('off')

    ax[1].imshow(image)
    ax[1].set_title(f'Now Image{idx+1}')
    ax[1].axis('off')
    plt.savefig(f'第{t+1}轮的图{idx+1}.png')
    plt.show()

def beta(timesteps,start=1e-4,end=0.02):#开始线性调度生成β 发现到后来loss增大 改为二次调度
    return torch.linspace(start**0.5,end**0.5,timesteps) ** 2

def forward_diffusion(x0,t,beta):
    batch_size=x0.size(0)
    noise=torch.randn_like(x0)
    a_t=torch.zeros(batch_size,1,1).to(x0.device)
    for i in range(batch_size):
        a_t[i]=torch.prod(1-beta[:t[i]])

    mean=torch.sqrt(a_t)*x0
    variance=torch.sqrt(1-a_t) * noise

    return mean+variance
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.input_to_hidden=nn.Linear(input_size,hidden_size)
        self.hidden_to_hidden=nn.Linear(hidden_size,hidden_size)
        self.hidden_to_output=nn.Linear(hidden_size,output_size* output_size)
        
    def forward(self,x,hidden):
        for t in range(x.size(1)):
            input_t=x[:,t,:]
            hidden=torch.relu(self.input_to_hidden(input_t)+self.hidden_to_hidden(hidden))
            output=self.hidden_to_output(hidden)
        output=output.view(-1,28,28)
        return output,hidden
class DDPM(nn.Module):
    def __init__(self,model,timesteps,betas):
        super(DDPM, self).__init__()
        self.model=model  
        self.timesteps=timesteps  
        self.betas=betas  

    def forward(self,x,hidden):
        for t in reversed(range(self.timesteps)):
            predicted_noise, hidden=self.model(x, hidden)
            beta_t=self.betas[t]
            x=(x-beta_t*predicted_noise)/torch.sqrt(1-beta_t)
        
        return x,hidden



def train_ddpm_rnn(model,dataset,optimizer,loss,betas,times,timesteps):
    train_losses=[]
    for i in range(times):
        
        total_loss=0.0

        for images,_ in dataset:
            images=images.view(-1,28,28)
            optimizer.zero_grad()
            t=torch.randint(0,timesteps,(images.size(0),)).to(images.device)
            noise=torch.randn_like(images)
            
            x_t=forward_diffusion(images, t, betas)

            x_t=x_t.view(-1,28,28)
            hidden=torch.zeros(images.size(0),model.model.hidden_to_hidden.out_features).to(images.device) 
            predicted_noise,hidden = model(x_t, hidden)
            los=loss(predicted_noise, noise)
            los.backward()
            optimizer.step()
            total_loss+=los.item()

        with torch.no_grad():
            picture,_=model(x_t,hidden)
        for idx in range(3):
            save_image(x_t[idx],picture[idx],i,idx)

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
    
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    
train_dataset=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)

train_loader=DataLoader(dataset=train_dataset,batch_size=256,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=256,shuffle=False)

input_size=28
hidden_size=128
output_size=28
timesteps=50
betas=beta(timesteps)


model=RNN(input_size,hidden_size,output_size)
ddpm_model=DDPM(model,timesteps,betas)
optimizer=optim.Adam(ddpm_model.parameters(),lr=0.00005)
loss=nn.MSELoss()
#使用均方误差损失函数
train_losses=train_ddpm_rnn(ddpm_model,train_loader,optimizer,loss,betas,times=10,timesteps=timesteps)
loss_show(train_losses)
#展示loss图像
