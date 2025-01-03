""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

from model.module.adapter_super import AdapterSuper

_logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=30, edgeitems=1)


class Mlp_SSF(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks, including SSF
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.drop2(x)
        return x

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_tiny_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_base_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_base_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch8_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_huge_patch14_224": _cfg(url=""),
    "vit_giant_patch14_224": _cfg(url=""),
    "vit_gigantic_patch14_224": _cfg(url=""),
    "vit_base2_patch32_256": _cfg(url="", input_size=(3, 256, 256), crop_pct=0.95),
    # patch models, imagenet21k (weights from official Google JAX impl)
    "vit_tiny_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_small_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_small_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_base_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_base_patch8_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        num_classes=21843,
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        num_classes=21843,
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
        num_classes=21843,
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
        hf_hub="timm/vit_huge_patch14_224_in21k",
        num_classes=21843,
    ),
    # SAM trained models (https://arxiv.org/abs/2106.01548)
    "vit_base_patch32_224_sam": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz"
    ),
    "vit_base_patch16_224_sam": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz"
    ),
    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    "vit_small_patch16_224_dino": _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
    ),
    "vit_small_patch8_224_dino": _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
    ),
    "vit_base_patch16_224_dino": _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
    ),
    "vit_base_patch8_224_dino": _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
    ),
    # deit models (FB weights)
    "deit_tiny_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_small_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    ),
    "deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
        classifier=("head", "head_dist"),
    ),
    # ViT ImageNet-21K-P pretraining by MILL
    "vit_base_patch16_224_miil_in21k": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
        num_classes=11221,
    ),
    "vit_base_patch16_224_miil": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm"
        "/vit_base_patch16_224_1k_miil_84_4.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
}

def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift

def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        LoRA_dim=1024,
        prefix_dim=1024,
        drop_rate_LoRA=0,
        add_lora_gate=False,
        add_prefix_gate=False,
        add_router=False,
        add_ssf=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.super_LoRA_dim = LoRA_dim

        # ssf setting 
        self.add_ssf = add_ssf
        if self.add_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim * 3)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

        # gate setting
        # self.add_lora_gate = add_lora_gate
        self.add_lora_gate = False
        self.add_prefix_gate = add_prefix_gate

        # router setting
        # self.add_router = add_router
        self.add_router = True
        
        self.at_num_module = 2  # LoRA , PATT (which is PA with Tanh)
        if self.add_router:
            self.router = nn.Sequential(
                nn.Linear(dim, self.at_num_module * num_heads * 3), # 3 for q,k,v or 2 for kv
                # nn.Sigmoid(),
            )
            # initialize rounter with mean 0 and std 0.02
            nn.init.normal_(self.router[0].weight, std=0.02, mean=0)

            # PATT Parallel Adapter setting
            self.Padapter = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim*3,
                reduction_dims=LoRA_dim,
                drop_rate_adapter=0.1,
                add_adapter_gate=False,
                sequential_adapter=False,
                parallel_adapter=True,
            )
        
        # LoRA init
        # print("LoRA_dim", LoRA_dim)
        if LoRA_dim > 0:
            self.LoRA_a = nn.Linear(dim, LoRA_dim, bias=False)
            nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            self.LoRA_b = nn.Linear(LoRA_dim, dim * 3, bias=False)
            nn.init.zeros_(self.LoRA_b.weight)

            if self.add_lora_gate:
                self.loRA_gate_q = nn.Linear(dim, num_heads)
                self.loRA_gate_k = nn.Linear(dim, num_heads)
                self.loRA_gate_v = nn.Linear(dim, num_heads)

                # initialize gate with mean 0 and std 0.02
                nn.init.normal_(self.loRA_gate_q.weight, std=0.02, mean=0)
                nn.init.normal_(self.loRA_gate_k.weight, std=0.02, mean=0)
                nn.init.normal_(self.loRA_gate_v.weight, std=0.02, mean=0)

        
        # prefix setting
        print("prefix_dim", prefix_dim)
        if prefix_dim > 0:
            self.prefix_tokens_key = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            self.prefix_tokens_value = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            # trunc_normal_(self.visual_prompt_token, std=.02)
            nn.init.xavier_uniform_(self.prefix_tokens_key)
            nn.init.xavier_uniform_(self.prefix_tokens_value)

            if self.add_prefix_gate:
                self.prefix_gate = nn.Linear(dim, 1)

                # initialize gate with mean 0 and std 0.02
                nn.init.normal_(self.prefix_gate.weight, std=0.02, mean=0)

        self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
        drop_rate_prefix = drop_rate_LoRA
        self.prefix_drop = nn.Dropout(p=drop_rate_prefix)

    def set_sample_config(self, sample_LoRA_dim, sample_prefix_dim):
        self.sample_LoRA_dim = sample_LoRA_dim
        self.LoRA_identity = False
        if self.sample_LoRA_dim == 0:
            self.LoRA_identity = True
        else:
            self.LoRA_a_weight = self.LoRA_a.weight[: self.sample_LoRA_dim, :]
            self.LoRA_b_weight = self.LoRA_b.weight[:, : self.sample_LoRA_dim]

        self.sample_prefix_dim = sample_prefix_dim
        self.prefix_identity = False
        if self.sample_prefix_dim == 0:
            self.prefix_identity = True
        else:
            self.prefix_weight_key = self.prefix_tokens_key[
                :, : self.sample_prefix_dim, :
            ]
            self.prefix_weight_value = self.prefix_tokens_value[
                :, : self.sample_prefix_dim, :
            ]
        if self.add_router:
            self.Padapter.set_sample_config(self.sample_LoRA_dim)

    def calc_sampled_param_num(self):
        if self.sample_LoRA_dim == 0:
            return 0
        else:
            return self.LoRA_a_weight.numel() + self.LoRA_b_weight.numel()

    def forward(self, x):
        B, N, C = x.shape

        if self.add_ssf:
            qkv = (
                ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            ) # 3 x B x num_heads x N x head_dim
        else:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            ) # 3 x B x num_heads x N x head_dim
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        if self.LoRA_identity == False:
            # PETL starts here
            # LoRA
            qkv_delta_lora = F.linear(self.LoRA_drop(x), self.LoRA_a_weight)
            qkv_delta_lora = (
                F.linear(qkv_delta_lora, self.LoRA_b_weight)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q_delta_lora, k_delta_lora, v_delta_lora = qkv_delta_lora.unbind(
                0
            )  # make torchscript happy (cannot use tensor as tuple)
            if self.add_router:
            # PATT Parallel Adapter
                qkv_delta_patt = self.Padapter(x)
                qkv_delta_patt = (
                    qkv_delta_patt.reshape(B, N, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q_delta_patt, k_delta_patt, v_delta_patt = qkv_delta_patt.unbind(
                    0
                )
            
            if self.add_lora_gate:

                # print("lora input shape",x.shape,)
                self.lora_scaling_q = torch.sigmoid(self.loRA_gate_q(x)) # shape (B,len_seq, num_heads)
                self.lora_scaling_k = torch.sigmoid(self.loRA_gate_k(x))
                self.lora_scaling_v = torch.sigmoid(self.loRA_gate_v(x))

                # reshape lora gate to (B, num_heads, len_seq,1)
                # self.lora_scaling = self.lora_scaling.unsqueeze(-1).transpose(1,2)
                self.lora_scaling_q = self.lora_scaling_q.unsqueeze(-1).transpose(1, 2)
                self.lora_scaling_k = self.lora_scaling_k.unsqueeze(-1).transpose(1, 2)
                self.lora_scaling_v = self.lora_scaling_v.unsqueeze(-1).transpose(1, 2)

                # mean gate for each gate
                # self.lora_scaling = self.lora_scaling.mean(dim=2).unsqueeze(-1) # shape (B,Head,1,1)
                self.lora_scaling_q = self.lora_scaling_q.mean(dim=2).unsqueeze(
                    -1
                )  # shape (B,Head,1,1)
                self.lora_scaling_k = self.lora_scaling_k.mean(dim=2).unsqueeze(
                    -1
                )  # shape (B,Head,1,1)
                self.lora_scaling_v = self.lora_scaling_v.mean(dim=2).unsqueeze(
                    -1
                )  # shape (B,Head,1,1)

                print(
                    "q lora_gate called with gate value",
                    self.lora_scaling_q.shape,
                    self.lora_scaling_q.max().item(),
                    self.lora_scaling_q.min().item(),
                    self.lora_scaling_q.squeeze(-1).squeeze(-1),
                )
                # print(
                #     "k lora_gate called with gate value",
                #     self.lora_scaling_k.shape,
                #     self.lora_scaling_k.max().item(),
                #     self.lora_scaling_k.min().item(),
                #     self.lora_scaling_k.squeeze(-1).squeeze(-1),
                # )
                # print(
                #     "v lora_gate called with gate value",
                #     self.lora_scaling_v.shape,
                #     self.lora_scaling_v.max().item(),
                #     self.lora_scaling_v.min().item(),
                #     self.lora_scaling_v.squeeze(-1).squeeze(-1),
                # )

                # print('lora_scaling shape',self.lora_scaling.shape, 'q_delta shape',q_delta.shape)
                q_delta, k_delta, v_delta = (
                    q_delta_lora * self.lora_scaling_q,
                    k_delta_lora * self.lora_scaling_k,
                    v_delta_lora * self.lora_scaling_v,
                )
                
            if self.add_router:
                self.router_scaling = self.router(x) #shape (B,len_seq, num_module*num_heads*3)
                # reshape softmax only on num_module
                # self.router_scaling = self.router_scaling.reshape(B, N, 3, self.num_heads, self.at_num_module).softmax(dim=-1)
                self.router_scaling = self.router_scaling.reshape(B, N, 3, self.num_heads, self.at_num_module).sigmoid()
                
                
                router_scaling_lora = self.router_scaling[:, :, :, : ,0] # shape (B, len_seq, 3, num_heads)
                router_scaling_patt = self.router_scaling[:, :, :, : ,1] 
                
                # reshape back to (B, len_seq, num_heads*3)
                # self.router_scaling = self.router_scaling.reshape(B, N, self.num_heads * 3 *  self.at_num_module)
                # # seperate router scaling for each module (LoRA, PATT)
                # router_scaling_lora = self.router_scaling[:, :, : self.num_heads * 3]
                # router_scaling_patt = self.router_scaling[:, :,  self.num_heads * 3 :]                
                
                # reshape router scaling to (B, num_heads, len_seq, 3)
                router_scaling_lora = router_scaling_lora.reshape(B, N, self.num_heads, 3).transpose(1, 2)
                router_scaling_patt = router_scaling_patt.reshape(B, N, self.num_heads, 3).transpose(1, 2)
                
                # mean gate for each gate
                # router_scaling_lora = router_scaling_lora.mean(dim=2, keepdim=True) # shape (B,Head,1,3)
                # router_scaling_patt = router_scaling_patt.mean(dim=2, keepdim=True)
                
                # print('SOFTMAX rounter scailing shape ', router_scaling_lora[:, :, :, 0].shape, "q_delta_lora shape", q_delta_lora.shape)
                # print("router scailing lora for q", router_scaling_lora[:, :, :, 0].max().item(), router_scaling_lora[:, :, :, 0].min().item(), router_scaling_lora[:, :, :, 0].squeeze(-1).squeeze(-1))
                # print("router scailing patt for q", router_scaling_patt[:, :, :, 0].max().item(), router_scaling_patt[:, :, :, 0].min().item(), router_scaling_patt[:, :, :, 0].squeeze(-1).squeeze(-1))
                print("router scailing lora for k", router_scaling_lora[:, :, :, 1].max().item(), router_scaling_lora[:, :, :, 1].min().item(), router_scaling_lora[:, :, :, 1].squeeze(-1).squeeze(-1))
                print("router scailing patt for k", router_scaling_patt[:, :, :, 1].max().item(), router_scaling_patt[:, :, :, 1].min().item(), router_scaling_patt[:, :, :, 1].squeeze(-1).squeeze(-1))
                # print("router scailing lora for v", router_scaling_lora[:, :, :, 2].max().item(), router_scaling_lora[:, :, :, 2].min().item(), router_scaling_lora[:, :, :, 2].squeeze(-1).squeeze(-1))
                # print("router scailing patt for v", router_scaling_patt[:, :, :, 2].max().item(), router_scaling_patt[:, :, :, 2].min().item(), router_scaling_patt[:, :, :, 2].squeeze(-1).squeeze(-1))
                q_delta_lora, k_delta_lora, v_delta_lora = (
                    q_delta_lora * router_scaling_lora[:, :, :, 0].unsqueeze(-1),
                    k_delta_lora * router_scaling_lora[:, :, :, 1].unsqueeze(-1),
                    v_delta_lora * router_scaling_lora[:, :, :, 2].unsqueeze(-1),
                )
                
                q_delta_patt, k_delta_patt, v_delta_patt = (
                    q_delta_patt * router_scaling_patt[:, :, :, 0].unsqueeze(-1),
                    k_delta_patt * router_scaling_patt[:, :, :, 1].unsqueeze(-1),
                    v_delta_patt * router_scaling_patt[:, :, :, 2].unsqueeze(-1),
                )
                
                q_delta, k_delta, v_delta = (
                    q_delta_lora, # q_delta_patt
                    k_delta_lora + k_delta_patt,
                    v_delta_lora + v_delta_patt,
                )
                    
            q, k, v = q + q_delta, k + k_delta, v + v_delta
            
        if self.prefix_identity == False:
            prefix_weight_key = self.prefix_weight_key.expand(B, -1, -1)
            prefix_weight_value = self.prefix_weight_value.expand(B, -1, -1)
            if self.add_prefix_gate:
                # print("prefix_gate called")
                self.prefix_scaling = torch.sigmoid(self.prefix_gate(x))
                prefix_weight_key, prefix_weight_value = (
                    prefix_weight_key * self.prefix_scaling,
                    prefix_weight_value * self.prefix_scaling,
                )
            k, v = torch.cat((k, prefix_weight_key), dim=1), torch.cat(
                (v, prefix_weight_value), dim=1
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.add_ssf:
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        visual_prompt_dim=1024,
        LoRA_dim=1024,
        adapter_dim=1024,
        prefix_dim=1024,
        drop_rate_LoRA=0,
        drop_rate_prompt=0,
        drop_rate_adapter=0,
        add_vpt_gate=False,
        add_adapter_gate=False,
        add_lora_gate=False,
        add_prefix_gate=False,
        sequential_adapter=False,
        parallel_adapter=False,
        current_layer=0,
        add_router=False,
        add_ssf=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            LoRA_dim=LoRA_dim,
            prefix_dim=prefix_dim,
            drop_rate_LoRA=drop_rate_LoRA,
            add_lora_gate=add_lora_gate,
            add_prefix_gate=add_prefix_gate,
            add_router=add_router,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if add_ssf:
             self.mlp = Mlp_SSF(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
             )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            

        # gate setting
        self.add_vpt_gate = add_vpt_gate
        # self.add_adapter_gate = add_adapter_gate
        self.add_adapter_gate = False

        # router setting
        self.ffn_num_module = 6  # LoRA, PA, SA () #9
        # self.add_router = add_router
        self.add_router = True
        if self.add_router:
            self.router = nn.Sequential(
                nn.Linear(dim, self.ffn_num_module),
                # nn.Softmax(dim=-1),
                # or sigmoid
                nn.Sigmoid(),
            )
            # initialize rounter with mean 0 and std 0.02
            nn.init.normal_(self.router[0].weight, std=0.02, mean=0)
            # adapter setting
            # self.Sadapter = AdapterSuper(
            #     embed_dims_in=dim,
            #     embed_dims_out=dim,
            #     reduction_dims=adapter_dim,
            #     drop_rate_adapter=drop_rate_adapter,
            #     add_adapter_gate=False,
            #     sequential_adapter=True,
            #     parallel_adapter=False,
            # )
            
            # self.Padapter = AdapterSuper(
            #     embed_dims_in=dim,
            #     embed_dims_out=dim,
            #     reduction_dims=adapter_dim,
            #     drop_rate_adapter=drop_rate_adapter,
            #     add_adapter_gate=False,
            #     sequential_adapter=False,
            #     parallel_adapter=True,
            # )
            
            # # LoRA setting

            # self.LoRA_a = nn.Linear(dim, adapter_dim, bias=False)
            # nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            # self.LoRA_b = nn.Linear(adapter_dim, dim, bias=False)
            # nn.init.zeros_(self.LoRA_b.weight)
            # self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
            
            self.Sadapter_10 = AdapterSuper(
                    embed_dims_in=dim,
                    embed_dims_out=dim,
                    reduction_dims=10,
                    drop_rate_adapter=drop_rate_adapter,
                    add_adapter_gate=False,
                    sequential_adapter=True,
                    parallel_adapter=False,
                )
                
            self.Padapter_10 = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=10,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=False,
                sequential_adapter=False,
                parallel_adapter=True,
            )
            
            # LoRA setting
            self.LoRA_a_10 = nn.Linear(dim, 10, bias=False)
            nn.init.kaiming_uniform_(self.LoRA_a_10.weight, a=math.sqrt(5))
            self.LoRA_b_10 = nn.Linear(10, dim, bias=False)
            nn.init.zeros_(self.LoRA_b_10.weight)
            self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
            
            self.Sadapter_5 = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=5,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=False,
                sequential_adapter=True,
                parallel_adapter=False,
            )
            
            self.Padapter_5 = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=5,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=False,
                sequential_adapter=False,
                parallel_adapter=True,
            )
            
            # LoRA setting
            self.LoRA_a_5 = nn.Linear(dim, 5, bias=False)
            nn.init.kaiming_uniform_(self.LoRA_a_5.weight, a=math.sqrt(5))
            self.LoRA_b_5 = nn.Linear(5, dim, bias=False)
            nn.init.zeros_(self.LoRA_b_5.weight)
            self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
            
            self.Sadapter_1 = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=1,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=False,
                sequential_adapter=True,
                parallel_adapter=False,
            )
            
            self.Padapter_1 = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=1,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=False,
                sequential_adapter=False,
                parallel_adapter=True,
            )
            
            # LoRA setting
            self.LoRA_a_1 = nn.Linear(dim, 1, bias=False)
            nn.init.kaiming_uniform_(self.LoRA_a_5.weight, a=math.sqrt(5))
            self.LoRA_b_1 = nn.Linear(1, dim, bias=False)
            nn.init.zeros_(self.LoRA_b_5.weight)
            self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
            
            
            
            

        # adapter setting
        # self.sequential_adapter = sequential_adapter
        self.sequential_adapter = True
        self.parallel_adapter = parallel_adapter
        if self.add_adapter_gate:
            self.adapter = AdapterSuper(
                embed_dims_in=dim,
                embed_dims_out=dim,
                reduction_dims=adapter_dim,
                drop_rate_adapter=drop_rate_adapter,
                add_adapter_gate=True,
                sequential_adapter=True,
                parallel_adapter=False,
            )

        # layer note
        self.current_layer = current_layer

        # prompt setting
        self.super_visual_prompt_dim = visual_prompt_dim
        print("visual_prompt_dim", visual_prompt_dim)
        if visual_prompt_dim > 0:
            self.visual_prompt_token = nn.Parameter(
                torch.zeros(1, visual_prompt_dim, dim)
            )
            # trunc_normal_(self.visual_prompt_token, std=.02)
            nn.init.xavier_uniform_(self.visual_prompt_token)
            if self.add_vpt_gate:
                print("vpt_gate initialized")
                self.vpt_gate = nn.Linear(dim, 1)
                # initialize gate with mean 0 and std 0.02
                nn.init.normal_(self.vpt_gate.weight, std=0.02, mean=0)

        self.drop_prompt = nn.Dropout(p=drop_rate_prompt)

        
        

    def set_sample_config(
        self,
        sample_LoRA_dim=None,
        sample_adapter_dim=None,
        sample_prefix_dim=None,
        sample_last_prompt_tuning_dim=None,
        sample_prompt_tuning_dim=None,
    ):

        self.attn.set_sample_config(
            sample_LoRA_dim=sample_LoRA_dim, sample_prefix_dim=sample_prefix_dim
        )
        

        # prompt setting
        self.sample_visual_prompt_dim = 0
        self.sample_last_prompt_tuning_dim = 0
        if self.super_visual_prompt_dim > 0:
            self.sample_visual_prompt_dim = sample_prompt_tuning_dim
            self.sample_last_prompt_tuning_dim = sample_last_prompt_tuning_dim

        self.sample_adapter_dim = sample_adapter_dim
        if self.add_adapter_gate:
            self.adapter.set_sample_config(sample_embed_dim=self.sample_adapter_dim)
        if self.add_router:
            # self.Sadapter.set_sample_config(sample_embed_dim=self.sample_adapter_dim)
            # self.Padapter.set_sample_config(sample_embed_dim=self.sample_adapter_dim)
            # self.LoRA_a_weight = self.LoRA_a.weight[: self.sample_adapter_dim, :]
            # self.LoRA_b_weight = self.LoRA_b.weight[:, : self.sample_adapter_dim]
            
            self.Sadapter_10.set_sample_config(sample_embed_dim=10) 
            self.Padapter_10.set_sample_config(sample_embed_dim=10)
            self.LoRA_a_weight_10 = self.LoRA_a_10.weight[:10, :] 
            self.LoRA_b_weight_10 = self.LoRA_b_10.weight[:, :10]
            self.Sadapter_5.set_sample_config(sample_embed_dim=5)
            self.Padapter_5.set_sample_config(sample_embed_dim=5)
            self.LoRA_a_weight_5 = self.LoRA_a_5.weight[:5, :]
            self.LoRA_b_weight_5 = self.LoRA_b_5.weight[:, :5]
            self.Sadapter_1.set_sample_config(sample_embed_dim=1) 
            self.Padapter_1.set_sample_config(sample_embed_dim=1)
            self.LoRA_a_weight_1 = self.LoRA_a_1.weight[:1, :]
            self.LoRA_b_weight_1 = self.LoRA_b_1.weight[:, :1]
        

    def calc_sampled_param_num(self):
        if self.sample_visual_prompt_dim != 0:
            sample_visual_prompt_param = self.visual_prompt_token[
                :, : self.sample_visual_prompt_dim, :
            ].numel()
        else:
            sample_visual_prompt_param = 0
        return sample_visual_prompt_param

    def forward(self, x):
        B = x.shape[0]
        if self.sample_visual_prompt_dim != 0:
            visual_prompt_tokens = self.visual_prompt_token[
                :, : self.sample_visual_prompt_dim, :
            ].expand(B, -1, -1)

            visual_prompt_tokens = self.drop_prompt(visual_prompt_tokens)
            if self.sample_last_prompt_tuning_dim == 0:
                if self.add_vpt_gate:

                    self.vpt_scaling = torch.sigmoid(self.vpt_gate(x))
                    self.vpt_scaling = self.vpt_scaling.mean(dim=1).unsqueeze(-1)
                    print(
                        self.current_layer,
                        "layer: \n vpt_gate called with value",
                        self.vpt_scaling.shape,
                        visual_prompt_tokens.shape,
                        self.vpt_scaling.max().item(),
                        self.vpt_scaling.min().item(),
                        self.vpt_scaling.squeeze(),
                    )
                    visual_prompt_tokens = visual_prompt_tokens * self.vpt_scaling
                x = torch.cat((x, visual_prompt_tokens), dim=1)

            else:
                if self.add_vpt_gate:

                    self.vpt_scaling = torch.sigmoid(self.vpt_gate(x))
                    print("vpt gate before mean", self.vpt_scaling.shape)
                    self.vpt_scaling = self.vpt_scaling.mean(dim=1).unsqueeze(-1)
                    print(
                        self.current_layer,
                        "layer: \n vpt_gate called with value",
                        self.vpt_scaling.shape,
                        visual_prompt_tokens.shape,
                        self.vpt_scaling.max().item(),
                        self.vpt_scaling.min().item(),
                        self.vpt_scaling.squeeze(),
                    )
                    print("visual_prompt_tokens shape", visual_prompt_tokens.shape, "vpt_scaling shape", self.vpt_scaling.shape)
                    visual_prompt_tokens = visual_prompt_tokens * self.vpt_scaling

                x = torch.cat(
                    (
                        x[:, : -self.sample_last_prompt_tuning_dim, :],
                        visual_prompt_tokens,
                    ),
                    dim=1,
                )

        x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.add_router:
            x_before_ffn = x
            x_after_ffn = self.drop_path(self.mlp(self.norm2(x_before_ffn)))
            router_scaling = self.router(x_before_ffn) #shape (B,len_seq, num_module)
            # comment out mean for softmax
            router_scaling = router_scaling.mean(dim=1, keepdim=True) # shape (B,1,num_module)
            # delta_sa = self.Sadapter(x_after_ffn)
            # delta_pa = self.Padapter(x)
            # delta_lora= F.linear(self.LoRA_drop(x), self.LoRA_a_weight)
            # delta_lora = F.linear(delta_lora, self.LoRA_b_weight)
            
            delta_sa_10 = self.Sadapter_10(x_after_ffn)
            delta_pa_10 = self.Padapter_10(x)
            # delta_lora_10= F.linear(self.LoRA_drop(x), self.LoRA_a_weight_10)
            # delta_lora_10 = F.linear(delta_lora_10, self.LoRA_b_weight_10)
            
            delta_sa_5 = self.Sadapter_5(x_after_ffn)
            delta_pa_5 = self.Padapter_5(x)
            # delta_lora_5= F.linear(self.LoRA_drop(x), self.LoRA_a_weight_5)
            # delta_lora_5 = F.linear(delta_lora_5, self.LoRA_b_weight_5)
            
            delta_sa_1 = self.Sadapter_1(x_after_ffn)
            delta_pa_1 = self.Padapter_1(x)
            # delta_lora_1= F.linear(self.LoRA_drop(x), self.LoRA_a_weight_1)
            # delta_lora_1 = F.linear(delta_lora_1, self.LoRA_b_weight_1)
            
            
            
            # delta_sa, delta_pa, delta_lora = (
            #     delta_sa * router_scaling[:, :, 0].unsqueeze(-1),
            #     delta_pa * router_scaling[:, :, 1].unsqueeze(-1),
            #     delta_lora * router_scaling[:, :, 2].unsqueeze(-1),
            # )
            
            # delta_sa_10, delta_pa_10, delta_lora_10 = (
            #     delta_sa_10 * router_scaling[:, :, 0].unsqueeze(-1),
            #     delta_pa_10 * router_scaling[:, :, 1].unsqueeze(-1),
            #     delta_lora_10 * router_scaling[:, :, 2].unsqueeze(-1),
            # )

            # delta_sa_5, delta_pa_5, delta_lora_5 = (
            #     delta_sa_5 * router_scaling[:, :, 3].unsqueeze(-1),
            #     delta_pa_5 * router_scaling[:, :, 4].unsqueeze(-1),
            #     delta_lora_5 * router_scaling[:, :, 5].unsqueeze(-1),
            # )

            # delta_sa_1, delta_pa_1, delta_lora_1 = (
            #     delta_sa_1 * router_scaling[:, :, 6].unsqueeze(-1),
            #     delta_pa_1 * router_scaling[:, :, 7].unsqueeze(-1),
            #     delta_lora_1 * router_scaling[:, :, 8].unsqueeze(-1),
            # )
            
            delta_sa_10, delta_pa_10 = (
                delta_sa_10 * router_scaling[:, :, 0].unsqueeze(-1),
                delta_pa_10 * router_scaling[:, :, 1].unsqueeze(-1),
            )

            delta_sa_5, delta_pa_5 = (
                delta_sa_5 * router_scaling[:, :, 2].unsqueeze(-1),
                delta_pa_5 * router_scaling[:, :, 3].unsqueeze(-1),
            )

            delta_sa_1, delta_pa_1 = (
                delta_sa_1 * router_scaling[:, :, 4].unsqueeze(-1),
                delta_pa_1 * router_scaling[:, :, 5].unsqueeze(-1),
            )
            
            

            
            print("rounter scailing for delta_sa_10", delta_sa_10.shape, router_scaling[:, :, 0].unsqueeze(-1).shape, router_scaling[:, :, 0].max().item(), router_scaling[:, :, 0].min().item(), router_scaling[:, :, 0].squeeze(-1).squeeze(-1))
            # print("rounter scailing for sa", delta_sa.shape, router_scaling[:, :, 0].unsqueeze(-1).shape, router_scaling[:, :, 0].max().item(), router_scaling[:, :, 0].min().item(), router_scaling[:, :, 0].squeeze(-1).squeeze(-1))
            # print("rounter scailing for pa", router_scaling[:, :, 1].max().item(), router_scaling[:, :, 1].min().item(), router_scaling[:, :, 1].squeeze(-1).squeeze(-1))
            # print("rounter scailing for lora", router_scaling[:, :, 2].max().item(), router_scaling[:, :, 2].min().item(), router_scaling[:, :, 2].squeeze(-1).squeeze(-1))
            # x = x_before_ffn + x_after_ffn + delta_sa + delta_pa + delta_lora
            
            # x = x_before_ffn + x_after_ffn + delta_sa_10 + delta_pa_10 + delta_lora_10
            # x += delta_sa_5 + delta_pa_5 + delta_lora_5
            # x += delta_sa_1 + delta_pa_1 + delta_lora_1
            
            x = x_before_ffn + x_after_ffn + delta_sa_10 + delta_pa_10 
            x += delta_sa_5 + delta_pa_5 
            x += delta_sa_1 + delta_pa_1
            
        elif self.parallel_adapter:
            delta_adapter = self.adapter(x)
            x = x + self.drop_path(self.mlp(self.norm2(x))) + delta_adapter
        else:
            # sequntial adapter
            x_after_ffn = self.drop_path(self.mlp(self.norm2(x)))
            x = x + x_after_ffn +self.adapter(x_after_ffn)

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        super_prompt_tuning_dim=1024,
        super_LoRA_dim=1024,
        super_adapter_dim=1024,
        super_prefix_dim=1024,
        drop_rate_LoRA=0,
        drop_rate_prompt=0,
        drop_rate_adapter=0,
        IS_not_position_VPT=False,
        add_vpt_gate=False,
        add_adapter_gate=False,
        add_lora_gate=False,
        add_prefix_gate=False,
        sequential_adapter=False,
        parallel_adapter=False,
        add_router=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.add_router = add_router
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    visual_prompt_dim=super_prompt_tuning_dim,
                    LoRA_dim=super_LoRA_dim,
                    adapter_dim=super_adapter_dim,
                    prefix_dim=super_prefix_dim,
                    drop_rate_LoRA=drop_rate_LoRA,
                    drop_rate_prompt=drop_rate_prompt,
                    drop_rate_adapter=drop_rate_adapter,
                    add_vpt_gate=add_vpt_gate,
                    add_adapter_gate=add_adapter_gate,
                    add_lora_gate=add_lora_gate,
                    add_prefix_gate=add_prefix_gate,
                    sequential_adapter=sequential_adapter,
                    parallel_adapter=parallel_adapter,
                    current_layer=i,
                    add_router=add_router,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        # For visual prompt tuning
        self.IS_not_position_VPT = IS_not_position_VPT
        self.super_prompt_tuning_dim = super_prompt_tuning_dim
        if super_prompt_tuning_dim > 0:
            self.visual_prompt_token = nn.Parameter(
                torch.zeros(1, super_prompt_tuning_dim, embed_dim)
            )
            nn.init.xavier_uniform_(self.visual_prompt_token)
            if not IS_not_position_VPT:
                self.visual_prompt_token_pos_embed = nn.Parameter(
                    torch.zeros(1, super_prompt_tuning_dim, embed_dim)
                )
                nn.init.xavier_uniform_(self.visual_prompt_token_pos_embed)

        self.drop_prompt = nn.Dropout(p=drop_rate_prompt)

        self.super_LoRA_dim = super_LoRA_dim
        self.super_adapter_dim = super_adapter_dim
        self.super_prefix_dim = super_prefix_dim

        self.init_weights(weight_init)

        self.freeze_stages()

    def freeze_stages(self):

        print("drop_rate", self.drop_rate)

        print("attn_drop_rate", self.attn_drop_rate)

        self.pos_drop.eval()
        self.patch_embed.eval()

        for block in self.blocks:
            block.eval()
            if self.super_LoRA_dim > 0:
                # attn lora setting
                block.attn.LoRA_a.train()
                block.attn.LoRA_b.train()
                block.attn.LoRA_drop.train()
                
                # attn PATT setting
                # if self.add_router:
                #     block.attn.Padapter.  train()
                
            block.drop_prompt.train()

            # if self.super_adapter_dim > 0:
            #     if self.add_router:
            #         block.Padapter.train()
            #         block.Sadapter.train()
                    
            #         # ffn lora setting
            #         block.LoRA_a.train()
            #         block.LoRA_b.train()
            #     else:
            #         block.adapter.train()

            if self.super_prefix_dim > 0:
                block.attn.prefix_drop.train()

        for name, param in self.named_parameters():
            if (
                "adapter" not in name
                and "prompt" not in name
                and "LoRA" not in name
                # and "prefix" not in name
                and "head" not in name
                and "gate" not in name
                and "router" not in name
                and "ssf" not in name
            ):
                param.requires_grad = False

        for m in self.modules():
            print(m.__class__.__name__, m.training)

        total_para_nums = 0
        adapter_para_nums = 0
        LoRA_para_nums = 0
        vp_para_nums = 0
        head_para_nums = 0
        gate_para_nums = 0
        router_para_nums = 0
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad:
                total_para_nums += param.numel()
                if "adapter" in name:
                    adapter_para_nums += param.numel()
                elif "LoRA" in name:
                    LoRA_para_nums += param.numel()
                elif "prompt" in name:
                    vp_para_nums += param.numel()
                elif "head" in name:
                    head_para_nums += param.numel()
                elif "gate" in name:
                    gate_para_nums += param.numel()
                elif "router" in name:
                    router_para_nums += param.numel()
                    

        print(
            "parameters:",
            total_para_nums,
            "adapter",
            adapter_para_nums,
            "LoRA",
            LoRA_para_nums,
            "prompt",
            vp_para_nums,
            "head",
            head_para_nums,
        )

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            # leave cls token as zeros to match jax impl
            named_apply(
                partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self
            )
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        if self.num_tokens == 2:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        total_para_nums = 0
        adapter_para_nums = 0
        LoRA_para_nums = 0
        vp_para_nums = 0
        head_para_nums = 0
        prefix_para_nums = 0
        model_para_nums = 0
        gate_para_nums = 0
        router_para_nums = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_para_nums += param.numel()
                if "adapter" in name:
                    adapter_para_nums += param.numel()
                elif "LoRA" in name:
                    LoRA_para_nums += param.numel()
                elif "prompt" in name:
                    vp_para_nums += param.numel()
                elif "head" in name:
                    head_para_nums += param.numel()
                elif "prefix" in name:
                    prefix_para_nums += param.numel()
                elif "gate" in name:
                    gate_para_nums += param.numel()
                elif "router" in name:
                    router_para_nums += param.numel()
            else:
                if "prefix" in name:
                    prefix_para_nums += param.numel()
                else:
                    model_para_nums += param.numel()

        print(
            "parameters:",
            total_para_nums,
            "adapter",
            adapter_para_nums,
            "LoRA",
            LoRA_para_nums,
            "prompt",
            vp_para_nums,
            "prefix",
            prefix_para_nums,
            "head",
            head_para_nums,
            "model",
            model_para_nums,
        )

    def set_sample_config(self, config: dict):

        # visual prompt tuning
        self.sample_prompt_tuning_dim = config["visual_prompt_dim"]

        # LoRA
        self.sample_LoRA_dim = config["lora_dim"]

        # Adapter
        self.sample_adapter_dim = config["adapter_dim"]

        # prefix_tuning
        self.sample_prefix_dim = config["prefix_dim"]

        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            blocks.set_sample_config(
                sample_prompt_tuning_dim=self.sample_prompt_tuning_dim[i],
                sample_LoRA_dim=self.sample_LoRA_dim[i],
                sample_prefix_dim=self.sample_prefix_dim[i],
                sample_last_prompt_tuning_dim=(
                    self.sample_prompt_tuning_dim[i - 1]
                    if i > 0
                    else self.sample_prompt_tuning_dim[0]
                ),
                sample_adapter_dim=self.sample_adapter_dim[i],
            )

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, "calc_sampled_param_num"):
                numels.append(module.calc_sampled_param_num())
        if self.sample_prompt_tuning_dim[0] == 0:
            sample_prompt_tuning_param = 0
        else:
            if not self.IS_not_position_VPT:
                sample_prompt_tuning_param = (
                    self.visual_prompt_token[
                        :, : self.sample_prompt_tuning_dim[0], :
                    ].numel()
                    + self.visual_prompt_token_pos_embed.numel()
                )
            else:
                sample_prompt_tuning_param = self.visual_prompt_token[
                    :, : self.sample_prompt_tuning_dim[0], :
                ].numel()

        return sum(numels) + sample_prompt_tuning_param

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat(
                (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1
            )
        x = self.pos_drop(x + self.pos_embed)

        if self.sample_prompt_tuning_dim[0] != 0:
            visual_prompt_tokens = self.visual_prompt_token[
                :, : self.sample_prompt_tuning_dim[0], :
            ].expand(B, -1, -1)
            if not self.IS_not_position_VPT:
                visual_prompt_tokens = (
                    visual_prompt_tokens
                    + self.visual_prompt_token_pos_embed[
                        :, : self.sample_prompt_tuning_dim[0], :
                    ]
                )
            visual_prompt_tokens = self.drop_prompt(visual_prompt_tokens)
            x = torch.cat((x, visual_prompt_tokens), dim=1)

        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(
    module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False
):
    """ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ""):
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    if (
        isinstance(model.head, nn.Linear)
        and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    ):
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
    if (
        isinstance(getattr(model.pre_logits, "fc", None), nn.Linear)
        and f"{prefix}pre_logits/bias" in w
    ):
        model.pre_logits.fc.weight.copy_(_n2p(w[f"{prefix}pre_logits/kernel"]))
        model.pre_logits.fc.bias.copy_(_n2p(w[f"{prefix}pre_logits/bias"]))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16) @ 384x384."""
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/32)"""
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/32) at 384x384."""
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base2_patch32_256(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32)
    # FIXME experiment
    """
    model_kwargs = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, **kwargs)
    model = _create_vision_transformer(
        "vit_base2_patch32_256", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights."""
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_huge_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560"""
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1408,
        mlp_ratio=48 / 11,
        depth=40,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_giant_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560"""
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1664,
        mlp_ratio=64 / 13,
        depth=48,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_gigantic_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548"""
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_sam", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224_sam(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548"""
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224_sam", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch16_224_dino(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294"""
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_patch8_224_dino(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294"""
    model_kwargs = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch8_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch8_224_dino(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294"""
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "deit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "deit_small_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil", pretrained=pretrained, **model_kwargs
    )
    return model
