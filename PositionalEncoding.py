import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1)
        p=10000**(-torch.arange(0,d_model,2).float()/d_model)
        pe[:,0::2]=torch.sin(pos*p)
        pe[:,1::2]=torch.cos(pos*p)
        self.pe=pe

    def forward(self,x):
        return x+self.pe[:x.size(1),:]
d_model = 8
max_len = 10
batch_size = 2
x = torch.randn(batch_size, max_len, d_model)
pe = PositionalEncoding(d_model, max_len)
output = pe(x)
print(output.shape)
