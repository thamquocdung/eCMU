import torch
import torch.nn as nn
import torch.nn.functional as F
from .conformer import ConformerBlock
import numpy as np
from torch import Tensor
from functools import partial

class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)        
        self.dim_f = dim_f
    
    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=False)
        x = x.permute([0,3,1,2])
        x = x.reshape([*batch_dims,c,2,-1,x.shape[-1]]).reshape([*batch_dims,c*2,-1,x.shape[-1]])
        return x[...,:self.dim_f,:]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c,f,t = x.shape[-3:]
        n = self.n_fft//2+1
        f_pad = torch.zeros([*batch_dims,c,n-f,t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims,c//2,2,n,t]).reshape([-1,2,n,t])
        x = x.permute([0,2,3,1])
        x = x[...,0] + x[...,1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims,2,-1])
        return x
    
def get_norm(norm_type):
    def norm(c, norm_type):   
        if norm_type=='BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type=='InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif norm_type=='LayerNorm':
            return LayerNorm(c, eps=1e-5)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()
    return partial(norm, norm_type=norm_type)

def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    elif act_type == "silu":
        return nn.SiLU()
    elif act_type == "prelu":
        return nn.PReLU()
    else:
        raise Exception
     
class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False),
            norm(out_c),
            act
        )
                                  
    def forward(self, x):
        return self.conv(x)

class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(   
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False),
            norm(out_c),
            act
        )
                                  
    def forward(self, x):
        return self.conv(x)

class D2_block(nn.Module):
    __constants__ = [
        'in_channels',
        'k',
        'L',
        'last_N',
    ]

    in_channels: int
    k: int
    L: int
    last_N: int

    def __init__(self,
                 in_channels,
                 act,
                 norm,
                 k=8,
                 L=5,
                 last_n_layers=3):
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.L = L
        self.last_N = last_n_layers

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(L):
            self.conv_layers.append(
                nn.Conv2d(
                    k if i > 0 else in_channels,
                    k * (L - i),
                    3,
                    padding=2 ** i,
                    dilation=2 ** i,
                    bias=False
                )
            )

            self.bn_layers.append(
                nn.Sequential(
                    norm(k),
                    act
                )
            )

    def get_output_channels(self):
        return self.k * min(self.L, self.last_N)

    def forward(self, input: torch.Tensor):
        # the input should be already BN + ReLU before
        outputs = []
        skips = []
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            tmp = conv(input).chunk(self.L - i, 1)
            input = tmp[0]
            tmp = tmp[1:]
            if i > 0:
                input = input + skips.pop(0)
                skips = [s + t for s, t in zip(skips, tmp)]
            else:
                skips = list(tmp)
            input = bn(input)
            outputs.append(input)

        assert len(skips) == 0
        if self.last_N > 1 and len(outputs) > 1:
            return torch.cat(outputs[-self.last_N:], 1)
        return outputs[-1]

class D3_block(nn.Module):
    def __init__(self,
                 in_channels,
                 M,
                 *args,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.d2_layers = nn.ModuleList()
        concat_channels = in_channels
        for i in range(M):
            self.d2_layers.append(
                D2_block(in_channels, *args, **kwargs)
            )
            # concat_channels = self.d2_layers[-1].get_output_channels()
            # in_channels += concat_channels

    def get_output_channels(self):
        return self.in_channels + sum(l.get_output_channels() for l in self.d2_layers)

    def forward1(self, input):
        raw_inputs = [input]
        for d2 in self.d2_layers:
            input = d2(torch.cat(raw_inputs, 1) if len(
                raw_inputs) > 1 else input)
            raw_inputs.append(input)

        return raw_inputs[-1] # torch.cat(raw_inputs, 1)
    
    def forward(self, x):
        for d2 in self.d2_layers:
            x = d2(x)
        return x

class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act, L=3):        
        super().__init__()
        self.block = nn.Module()
        
        self.block.tfc = nn.Sequential(
            D3_block(in_channels=in_c, M=3, act=act, norm=norm, L=L, k=c, last_n_layers=1),
            norm(in_c),
            act
        )
        self.block.tdf = nn.Sequential(
            nn.Linear(f, f//bn, bias=False),
            norm(c),
            act,
            nn.Linear(f//bn, f, bias=False),
            norm(c),
            act
        )
                 
    def forward(self, x):
        x = self.block.tfc(x)
        x = x + self.block.tdf(x)
        return x

class MultiHeadAttention(nn.Module):
    __constants__ = [
        'out_channels',
        'd_model',
        'n_heads',
        'query_shape',
        'memory_flange'
    ]

    max_bins: int
    d_model: int
    n_heads: int
    query_shape: int
    memory_flange: int

    def __init__(self, in_channels, out_channels, d_model=32, n_heads=8, query_shape=24, memory_flange=8):
        super().__init__()

        self.out_channels = out_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_shape = query_shape
        self.memory_flange = memory_flange

        self.qkv_conv = nn.Conv2d(in_channels, d_model * 3, 3, padding=1)
        self.out_conv = nn.Conv2d(
            d_model, out_channels, 3, padding=1, bias=False)

    def _pad_to_multiple_2d(self, x: torch.Tensor, query_shape: int):
        t = x.shape[-1]
        offset = t % query_shape
        if offset != 0:
            offset = query_shape - offset

        if offset > 0:
            return F.pad(x, [0, offset])
        return x

    def forward(self, x):
        qkv = self._pad_to_multiple_2d(self.qkv_conv(x), self.query_shape)
        qkv = qkv.view((qkv.shape[0], self.n_heads, -1) + qkv.shape[2:])
        q, k, v = qkv.chunk(3, 2)

        k_depth_per_head = self.d_model // self.n_heads
        q = q * k_depth_per_head ** -0.5

        k = F.pad(k, [self.memory_flange] * 2)
        v = F.pad(v, [self.memory_flange] * 2)

        unfold_q = q.reshape(
            q.shape[:4] + (q.shape[4] // self.query_shape, self.query_shape))
        unfold_k = k.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)
        unfold_v = v.unfold(-1, self.query_shape +
                            self.memory_flange * 2, self.query_shape)

        unfold_q = unfold_q.permute(0, 1, 4, 3, 5, 2)
        tmp = unfold_q.shape
        unfold_q = unfold_q.reshape(
            -1, unfold_q.shape[-2] * unfold_q.shape[-3], k_depth_per_head)
        unfold_k = unfold_k.permute(0, 1, 4, 2, 3, 5).reshape(
            unfold_q.shape[0], k_depth_per_head, -1)
        unfold_v = unfold_v.permute(0, 1, 4, 3, 5, 2).reshape(
            unfold_q.shape[0], -1, k_depth_per_head)

        bias = (unfold_k.abs().sum(-2, keepdim=True)
                == 0).to(unfold_k.dtype) * -1e-9
        # correct value should be -1e9, we type this by accident and use it during the whole competition
        # so just leave it what it was :)

        logits = unfold_q @ unfold_k + bias
        weights = logits.softmax(-1)
        out = weights @ unfold_v

        out = out.view(tmp).permute(0, 1, 5, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] *
                          out.shape[2], out.shape[3], -1)
        out = out[..., :x.shape[2], :x.shape[3]]

        return self.out_conv(out)

class Net(nn.Module):
    def __init__(self, sources, overlap, freq_emb=False):
        super().__init__()
        
        norm = get_norm(norm_type="BatchNorm")
        act = get_act(act_type="relu")

        
        self.num_target_instruments = len(sources)
        self.num_subbands = 1
        self.overlap = overlap
        dim_c = self.num_subbands * 2 * 2         
        n = 3
        scale = (2,2)
        l = 2
        c = 32
        g = 32
        bn = 8  
        dim_f = 2048
        f = dim_f // self.num_subbands
        dim_t = 128
        hop_length = 2048
        self.chunk_size = (dim_t-1)*hop_length
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim_c, out_channels=c, kernel_size=(1, 1)),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )

 
        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c+g, scale, norm, act) 
            f = f//scale[1]
            c += g
            self.encoder_blocks.append(block)                
               

        self.bottleneck_block = nn.Sequential(TSCB(num_channel=c), TSCB(num_channel=c))

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):                
            block = nn.Module()
            block.upscale = Upscale(c, c-g, scale, norm, act)
            f = f*scale[1]
            c -= g  
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block) 
              
        self.final_conv = nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)

        
        self.stft = STFT(n_fft=6144, hop_length=hop_length, dim_f=dim_f)
    
    def cac2cws(self, x):
        k = self.num_subbands
        b,c,f,t = x.shape
        x = x.reshape(b,c,k,f//k,t)
        x = x.reshape(b,c*k,f//k,t)
        return x
    
    def cws2cac(self, x):
        k = self.num_subbands
        b,c,f,t = x.shape
        x = x.reshape(b,c//k,k,f,t)
        x = x.reshape(b,c//k,f*k,t)
        return x
    
    def forward(self, x, return_spec=False):
        x = self.stft(x)
        mix_spec = x
        # mix = x = self.cac2cws(x)
        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1,-2)
        
        encoder_outputs = []
        for block in self.encoder_blocks:  
            x = block.tfc_tdf(x) 
            encoder_outputs.append(x)
            x = block.downscale(x)              

        x = self.bottleneck_block(x)
        for block in self.decoder_blocks:            
            x = block.upscale(x)
            x = x*encoder_outputs.pop()
            x = block.tfc_tdf(x) 
            
        x = x.transpose(-1,-2)
        
        # x = x * first_conv_out  # reduce artifacts
        x = self.final_conv(x) 
        
        # x = self.cws2cac(x)
        
        # if self.num_target_instruments > 1:
        b,c,f,t = x.shape
        x = x.reshape(b,self.num_target_instruments,-1,f,t)

        pred_spec = x
        target_wave_hat = self.stft.inverse(x)
        if return_spec:
            return target_wave_hat, pred_spec, mix_spec
        else:
            return target_wave_hat


if __name__ == "__main__":
    x = torch.rand(2,2,255*1024).cuda()
    net = Net(sources=["vocals", "drums", "bass"], overlap=0.5).cuda()
    print(net)
    print(sum(p.numel() for p in net.parameters()
              if p.requires_grad))
    y = net(x)
    print(y.shape)