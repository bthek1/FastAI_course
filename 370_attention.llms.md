# Attention

Self-attention and multi-head attention built from scratch

Author

Benedict Thekkel

``` python
import math,torch
from torch import nn
from fastAIcourse.activations import *
```

``` python
import matplotlib.pyplot as plt
```

``` python
from diffusers.models.attention import Attention as AttentionBlock
```

``` python
set_seed(42)
x = torch.randn(64,32,16,16)
```

``` python
t = x.view(*x.shape[:2], -1).transpose(1, 2)
t.shape
```

    torch.Size([64, 256, 32])

``` python
ni = 32
```

``` python
sk = nn.Linear(ni, ni)
sq = nn.Linear(ni, ni)
sv = nn.Linear(ni, ni)
```

``` python
k = sk(t)
q = sq(t)
v = sv(t)
```

``` python
(q@k.transpose(1,2)).shape
```

    torch.Size([64, 256, 256])

``` python
class SelfAttention(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.scale = math.sqrt(ni)
        self.norm = nn.GroupNorm(1, ni)
        self.q = nn.Linear(ni, ni)
        self.k = nn.Linear(ni, ni)
        self.v = nn.Linear(ni, ni)
        self.proj = nn.Linear(ni, ni)
    
    def forward(self, x):
        inp = x
        n,c,h,w = x.shape
        x = self.norm(x)
        x = x.view(n, c, -1).transpose(1, 2)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        x = self.proj(x)
        x = x.transpose(1,2).reshape(n,c,h,w)
        return x+inp
```

``` python
sa = SelfAttention(32)
```

``` python
ra = sa(x)
ra.shape
```

    torch.Size([64, 32, 16, 16])

``` python
ra[0,0,0]
```

    tensor([ 1.91,  1.42,  0.84, -2.16,  0.63, -1.24, -0.08, -1.68, -0.79,  1.61, -0.39, -1.43, -0.75, -0.60, -0.83,  0.75],
           grad_fn=<SelectBackward0>)

``` python
def cp_parms(a,b):
    b.weight = a.weight
    b.bias = a.bias
```

``` python
at = AttentionBlock(32, norm_num_groups=1)
src = sa.q,sa.k,sa.v,sa.proj,sa.norm
dst = at.query,at.key,at.value,at.proj_attn,at.group_norm
for s,d in zip(src,dst): cp_parms(s,d)
```

``` python
rb = at(x)
rb[0,0,0]
```

``` python
sqkv = nn.Linear(ni, ni*3)
st = sqkv(t)
st.shape
```

    torch.Size([64, 256, 96])

``` python
q,k,v = torch.chunk(st, 3, dim=-1)
q.shape
```

``` python
(k@q.transpose(1,2)).shape
```

``` python
class SelfAttention(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.scale = math.sqrt(ni)
        self.norm = nn.BatchNorm2d(ni)
        self.qkv = nn.Linear(ni, ni*3)
        self.proj = nn.Linear(ni, ni)
    
    def forward(self, inp):
        n,c,h,w = inp.shape
        x = self.norm(inp).view(n, c, -1).transpose(1, 2)
        q,k,v = torch.chunk(self.qkv(x), 3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        x = self.proj(x).transpose(1,2).reshape(n,c,h,w)
        return x+inp
```

``` python
class SelfAttention(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.scale = math.sqrt(ni)
        self.norm = nn.BatchNorm2d(ni)
        self.qkv = nn.Linear(ni, ni*3)
        self.proj = nn.Linear(ni, ni)
    
    def forward(self, x):
        x = self.norm(x).transpose(1, 2)
        q,k,v = torch.chunk(self.qkv(x), 3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        return self.proj(x).transpose(1,2)
```

``` python
sa = SelfAttention(32)
sa(x).shape
```

    torch.Size([64, 32, 16, 16])

``` python
sa(x).std()
```

    tensor(1.0047, grad_fn=<StdBackward0>)

``` python
def heads_to_batch(x, heads):
    n,sl,d = x.shape
    x = x.reshape(n, sl, heads, -1)
    return x.transpose(2, 1).reshape(n*heads,sl,-1)

def batch_to_heads(x, heads):
    n,sl,d = x.shape
    x = x.reshape(-1, heads, sl, d)
    return x.transpose(2, 1).reshape(-1,sl,d*heads)
```

``` python
from einops import rearrange
```

``` python
t2 = rearrange(t , 'n s (h d) -> (n h) s d', h=8)
t.shape, t2.shape
```

    (torch.Size([64, 256, 32]), torch.Size([512, 256, 4]))

``` python
t3 = rearrange(t2, '(n h) s d -> n s (h d)', h=8)
```

``` python
t2.shape,t3.shape
```

    (torch.Size([512, 256, 4]), torch.Size([64, 256, 32]))

``` python
(t==t3).all()
```

    tensor(True)

``` python
class SelfAttentionMultiHead(nn.Module):
    def __init__(self, ni, nheads):
        super().__init__()
        self.nheads = nheads
        self.scale = math.sqrt(ni/nheads)
        self.norm = nn.BatchNorm2d(ni)
        self.qkv = nn.Linear(ni, ni*3)
        self.proj = nn.Linear(ni, ni)
    
    def forward(self, inp):
        n,c,h,w = inp.shape
        x = self.norm(inp).view(n, c, -1).transpose(1, 2)
        x = self.qkv(x)
        x = rearrange(x, 'n s (h d) -> (n h) s d', h=self.nheads)
        q,k,v = torch.chunk(x, 3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1)@v
        x = rearrange(x, '(n h) s d -> n s (h d)', h=self.nheads)
        x = self.proj(x).transpose(1,2).reshape(n,c,h,w)
        return x+inp
```

``` python
sa = SelfAttentionMultiHead(32, 4)
sx = sa(x)
sx.shape
```

    torch.Size([64, 32, 16, 16])

``` python
sx.mean(),sx.std()
```

    (tensor(0.0248, grad_fn=<MeanBackward0>),
     tensor(1.0069, grad_fn=<StdBackward0>))

``` python
nm = nn.MultiheadAttention(32, num_heads=8, batch_first=True)
nmx,nmw = nm(t,t,t)
nmx = nmx+t
```

``` python
nmx.mean(),nmx.std()
```

    (tensor(-0.0021, grad_fn=<MeanBackward0>),
     tensor(1.0015, grad_fn=<StdBackward0>))

Back to top
