# SPDX-License-Identifier: Apache-2.0
# Formula follows the deployed SGLang/FLA layernorm_gated.py (Tri Dao, 2024;
# Hygon modifications, 2026). This is a standalone inference-only candidate.
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_gate(X,Z,W,Y,Q,S, z_token_stride,z_head_stride,eps,
               QUANT:tl.constexpr,EMIT_NORM:tl.constexpr):
    token=tl.program_id(0)
    heads=tl.arange(0,32)
    cols=tl.arange(0,128)
    mask=heads[:,None]<24
    off=token*3072+heads[:,None]*128+cols[None,:]
    x=tl.load(X+off,mask,other=0).to(tl.float32)
    variance=tl.sum(x*x,axis=1)/128
    rstd=tl.rsqrt(variance+eps)
    weight=tl.load(W+cols).to(tl.float32)
    y=(x*rstd[:,None])*weight[None,:]
    zoff=token*z_token_stride+heads[:,None]*z_head_stride+cols[None,:]
    z=tl.load(Z+zoff,mask,other=0).to(tl.float32)
    # Standard RMS weight (no Gemma +1). Native GDN keeps norm and gating in
    # FP32 until the final store; do not insert the MLP SiLU's extra BF16 cast.
    y=y*(z*tl.sigmoid(z))
    bf16_y=y.to(tl.bfloat16)
    if EMIT_NORM:
        tl.store(Y+off,bf16_y,mask)
    if QUANT:
        rounded=tl.where(mask,bf16_y.to(tl.float32),0.0)
        absmax=tl.maximum(tl.max(tl.max(tl.abs(rounded),axis=1),axis=0),1e-10)
        scale=absmax/127
        quant=libdevice.nearbyint(rounded*(127/absmax)).to(tl.int8)
        tl.store(Q+off,quant,mask)
        tl.store(S+token,scale)


def norm_gate(x,z,weight,*,quantize=True,emit_norm=False,num_warps=4,eps=1e-6):
    if x.dtype!=torch.bfloat16 or z.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16:
        raise ValueError('BF16 input/gate/standard weight required')
    if x.ndim!=2 or x.shape[1]!=3072 or not x.is_contiguous():
        raise ValueError('Expected contiguous [tokens,3072] core attention output')
    if z.shape!=(x.shape[0],24,128) or z.stride(-1)!=1 or weight.shape!=(128,):
        raise ValueError('Expected gate [tokens,24,128] and standard weight [128]')
    if not x.is_cuda or not (x.device==z.device==weight.device) or not weight.is_contiguous():
        raise ValueError('Same device and contiguous standard weight required')
    if num_warps not in (4,8):raise ValueError('Bounded 4/8 warp sweep only')
    norm=torch.empty_like(x) if emit_norm or not quantize else None
    q=torch.empty_like(x,dtype=torch.int8) if quantize else None
    scale=torch.empty(x.shape[0],1,device=x.device,dtype=torch.float32) if quantize else None
    if x.shape[0]:
        _norm_gate[(x.shape[0],)](x,z,weight,norm if norm is not None else x,
            q if q is not None else x,scale if scale is not None else x,
            z.stride(0),z.stride(1),eps,quantize,norm is not None,
            num_warps=num_warps,num_stages=1)
    return q,scale,norm
