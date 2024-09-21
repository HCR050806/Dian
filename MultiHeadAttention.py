import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_head == 0, "头数必须整除维数"
        self.d_model=d_model
        self.num_head=num_head
        self.depth=d_model//num_head
        self.linear_Q=nn.Linear(d_model,d_model)
        self.linear_K=nn.Linear(d_model,d_model)
        self.linear_V=nn.Linear(d_model,d_model)
        
        self.linear_end=nn.Linear(d_model,d_model)
        
    def multi_head(self,x,batch_size):
        x=x.view(batch_size,-1,self.num_head,self.depth)
        return x.permute(0,2,1,3)
    
    def Scaled_Dot_Product_Attention(self,q,k,v):
        matmul=torch.matmul(q, k.transpose(-2, -1))
        d_k=torch.tensor(k.size()[-1])
        scale=matmul/torch.sqrt(d_k)
        attention_weights=F.softmax(scale,dim=-1)
        scaled_attention=torch.matmul(attention_weights,v)
        return scaled_attention,attention_weights
    
    def concat(self,scaled_attention,batch_size):
        scaled_attention=scaled_attention.permute(0,2,1,3).contiguous()
        attention=scaled_attention.view(batch_size,-1,self.d_model)
        return attention
    
    def forward(self,q,k,v):
        batch_size=q.size(0)
        
        q=self.linear_Q(q)
        k=self.linear_K(k)
        v=self.linear_V(v)
        
        q=self.multi_head(q,batch_size)
        k=self.multi_head(k,batch_size)
        v=self.multi_head(v,batch_size)
        
        scaled_attention,attention_weights=self.Scaled_Dot_Product_Attention(q,k,v)
        
        attention=self.concat(scaled_attention,batch_size)
        
        attention=self.linear_end(attention)
        
        return attention,attention_weights
    
    

d_model=64
num_heads=8  
batch_size=1 
seq_len=10  


q = torch.rand(batch_size, seq_len, d_model)
k = torch.rand(batch_size, seq_len, d_model)
v = torch.rand(batch_size, seq_len, d_model)
#生成随机矩阵

multihead_attention = MultiHeadAttention(d_model, num_heads)


attention, attention_weights = multihead_attention(q, k, v)
torch.set_printoptions(sci_mode=False, precision=4)#运行后发现出现科学计数法 补一行代码规范化输出
print("注意力矩阵:", attention)
print("注意力权重:", attention_weights)
