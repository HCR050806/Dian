import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model=d_model
        self.max_len=max_len
        
        pos = torch.arange(max_len).unsqueeze(1)
        p=10000**(-torch.arange(0,d_model//2).float()/d_model).unsqueeze(0)#生成一系列θ角
        
        m8=pos*p
        
        expanded_m8=m8.repeat(1,2)
        
        self.cos_m8=torch.cos(expanded_m8)
        self.sin_m8=torch.sin(expanded_m8)
        
        
    def forward(self,x):
        len=x.size(1)
        cos_m8=self.cos_m8[:len,:]
        sin_m8=self.sin_m8[:len,:]
        
        x1=x[:,:,:self.d_model//2]
        x2=-x[:,:,self.d_model//2:]
        
        new_x=torch.cat([x2,x1],dim=2)
        x_rotated=x*cos_m8+new_x*sin_m8
        return x_rotated
d_model=8
max_len=10
batch_size=2
x = torch.randn(batch_size,max_len,d_model)

rpe=RotaryPositionalEncoding(d_model,max_len)
output=rpe(x)
print(output.shape) 
# 旋转位置编码和绝对位置编码的区别：
# 在绝对位置编码中，每个位置的编码是通过正弦和余弦函数计算得出的，得到的是一个独立于其他位置的编码。
# 而旋转位置编码不仅与位置本身有关，还与它们在序列中的相对位置关系有关。