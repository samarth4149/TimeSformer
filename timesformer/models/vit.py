# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_small_patch16_224': _cfg(
        url='https://no-default-model-url',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time',
                 st_adapter=False, sta_dim=384, first_patch_idx=1):
        super().__init__()
        self.attention_type = attention_type
        self.st_adapter = st_adapter
        self.act = act_layer()
        self.first_patch_idx = first_patch_idx
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        assert (not self.st_adapter) or (self.attention_type == 'divided_space_time'), \
            'st_adapter can only be used when attention_type is \'divided_space_time\''
        
        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
            if self.st_adapter:
                self.stadapter_w_down = nn.Linear(dim, sta_dim)
                self.stadapter_conv3d = nn.Conv3d(sta_dim, sta_dim, kernel_size=(1, 1, 3), padding=(0, 0, 1))
                self.stadapter_w_up = nn.Linear(sta_dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)        


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - self.first_patch_idx) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            if self.st_adapter:
                # 1. Multiply all tokens with W_down
                res = self.stadapter_w_down(x)
                # 2. Remove cls token and reshape to (B, C, H, W, T)
                cls_token = res[:,self.first_patch_idx-1:self.first_patch_idx,:]
                res = rearrange(res[:, self.first_patch_idx:, :], 'b (h w t) m -> b m h w t', b=B, h=H, w=W, t=T)
                # 3. Apply 3d conv
                res = self.stadapter_conv3d(res)
                # 4. Reshape to (B, H*W*T, C) and add back cls token
                res = rearrange(res, 'b m h w t -> b (h w t) m', b=B, h=H, w=W, t=T)
                res = torch.cat([cls_token, res], dim=1)
                # 5. Apply GeLU
                res = self.act(res)
                # 6. Multiply with W_up
                res = self.stadapter_w_up(res)
                x = x + res
                xt = x[:,self.first_patch_idx:,:] # name xt not quite apt, just using to resuse code below
            else:
                ## Temporal
                xt = x[:,self.first_patch_idx:,:]
                xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
                res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
                res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
                res_temporal = self.temporal_fc(res_temporal)
                xt = x[:,self.first_patch_idx:,:] + res_temporal

            ## Spatial
            if self.first_patch_idx == 1: #if cls token
                init_cls_token = x[:,self.first_patch_idx:self.first_patch_idx+1,:]
                cls_token = init_cls_token.repeat(1, T, 1)
                cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
                xs = xt
                xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
                xs = torch.cat((cls_token, xs), 1)
                res_spatial = self.drop_path(self.attn(self.norm1(xs)))

                ### Taking care of CLS token
                cls_token = res_spatial[:, self.first_patch_idx-1, :]
                cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
                cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
                res_spatial = res_spatial[:, self.first_patch_idx:, :]
                res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
                res = res_spatial
                x = xt

                ## Mlp
                x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else: # no cls token below
                assert self.first_patch_idx == 0, "No class token so first_patch_idx must be 0"
                xs = xt
                xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
                res_spatial = self.drop_path(self.attn(self.norm1(xs)))
                res_spatial = res_spatial[:, self.first_patch_idx:, :]
                res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
                res = res_spatial
                x = xt
                ## Mlp
                x = x + res
                x = x + self.drop_path(self.mlp(self.norm2(x)))


            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
        If tublet_dim=2 and patch_size=16. Patches become 2x16x16
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, tublet_dim=1, num_frames=8, use_omnivore_vit=False):

        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tublet_dim = tublet_dim
        self.num_frames = num_frames
        self.use_omnivore_vit = use_omnivore_vit

        if tublet_dim == 1:
            self.img_size = img_size
            self.patch_size = patch_size
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        else:
            self.img_size = (tublet_dim,) + img_size
            self.patch_size = (tublet_dim,) + patch_size
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if self.use_omnivore_vit:
            with torch.no_grad():
                dummy_img = torch.zeros(
                    (1, in_chans, num_frames, ) + img_size
                )
                self.patches_layout = tuple(self.proj(dummy_img).shape[2:])
                self.num_patches = np.prod(self.patches_layout)
        else:
            num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.num_patches = num_patches
            self.patches_layout = None

    def forward(self, x):
        B, C, T, H, W = x.shape
        if self.tublet_dim == 1:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.proj(x)
            W = x.size(-1)
            x = x.flatten(2).transpose(1, 2)
        else:
            x = self.proj(x)
            T = x.shape[2] #override T
            if not self.use_omnivore_vit:
                x = rearrange(x, 'b c t h w -> (b t) c h w')
            W = x.size(-1)
            x = x.flatten(2).transpose(1, 2) #shape of (b t) c (h w)
        return x, T, W

class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.,
                 st_adapter=False, st_adapter_dim=384, tublet_dim=1, use_time_embed=True, use_cls_token=True, use_omnivore_vit=False):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_omnivore_vit = use_omnivore_vit
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tublet_dim=tublet_dim,
            num_frames=num_frames, use_omnivore_vit=use_omnivore_vit)
        num_patches = self.patch_embed.num_patches
        assert num_frames % tublet_dim == 0, "num_frames must be divisible by tublet_dim"
        self.use_time_embed = use_time_embed
        self.use_cls_token = use_cls_token

        ## Positional Embeddings
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
            self.first_patch_idx = 1
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.first_patch_idx = 0
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only' and use_time_embed:
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames//tublet_dim, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type, st_adapter=st_adapter, sta_dim=st_adapter_dim,
                first_patch_idx=self.first_patch_idx)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        if self.use_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    '''
    below is omnivore implementation of pos embedding
    '''
    @classmethod
    def get_pos_embedding(
            cls,
            npatch_per_img,
            pos_embed,
            patches_layout,
            input_shape,
            first_patch_idx=1,
    ):
        pos_embed = cls.interpolate_pos_encoding(
            npatch_per_img,
            pos_embed,
            patches_layout,
            input_shape=input_shape,
            first_patch_idx=first_patch_idx,
        )
        return pos_embed
    @staticmethod
    def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
        N = pos_embed.shape[1]
        if N == target_spatial_size:
            return pos_embed
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(target_spatial_size / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    @classmethod
    def interpolate_pos_encoding(
        cls,
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=None,
        first_patch_idx=1,
    ):
        assert (
            first_patch_idx == 0 or first_patch_idx == 1
        ), "there is 1 CLS token or none"
        N = pos_embed.shape[1] - first_patch_idx  # since it's 1 if cls_token exists
        if npatch_per_img == N:
            return pos_embed
        class_emb = pos_embed[:, :first_patch_idx]
        pos_embed = pos_embed[:, first_patch_idx:]

        if input_shape is None or patches_layout[0] == 1:
            # simple 2D pos embedding, no temporal component
            pos_embed = cls.interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
        elif patches_layout[0] > 1:
            # pos embed has a temporal component
            assert len(input_shape) == 4, "temporal interpolation not supported"
            # we only support 2D interpolation in this case
            num_frames = patches_layout[0]
            num_spatial_tokens = patches_layout[1] * patches_layout[2]
            pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
            # interpolate embedding for zeroth frame
            pos_embed = cls.interpolate_pos_encoding_2d(
                npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0)
            )
        else:
            raise ValueError("This type of interpolation isn't implemented")

        return torch.cat((class_emb, pos_embed), dim=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.use_omnivore_vit:
            npatch_per_img = x.shape[1]
            input_shape = x.shape
            pos_embed = self.get_pos_embedding(
                npatch_per_img,
                self.pos_embed,
                self.patch_embed.patches_layout,
                input_shape,
                first_patch_idx=self.first_patch_idx,
            )
            x = x + pos_embed
        else:
            ## resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = self.pos_embed
                fpi = self.first_patch_idx
                cls_pos_embed = pos_embed[0,fpi-1:fpi,:].unsqueeze(0)
                other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,self.first_patch_idx:]
            if self.use_omnivore_vit:
                x = rearrange(x, 'b (n t) m -> (b n) t m', b=B, t=T)
            else:
                x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            if self.use_time_embed:
                ## Resizing time embeddings in case they don't match
                if T != self.time_embed.size(1):
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                    new_time_embed = new_time_embed.transpose(1, 2)
                    x = x + new_time_embed
                else:
                    x = x + self.time_embed
                x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame
        if self.use_cls_token:
            x = self.norm(x)
            return x[:, 0]
        else:
            x = x[:, self.first_patch_idx:, ...].mean(dim=1)
            x = self.norm(x)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            if len(v.shape) <= 4: # condition need to support 3d convolution where there's 5 dimensions
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_small_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_small_patch16_224, self).__init__()
        self.pretrained=cfg.MODEL.PRETRAINED
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, st_adapter=cfg.TIMESFORMER.ST_ADAPTER, st_adapter_dim=cfg.TIMESFORMER.ST_ADAPTER_DIM, tublet_dim=cfg.TIMESFORMER.TUBLET_DIM, use_time_embed=cfg.TIMESFORMER.USE_TIME_EMBED, use_cls_token=cfg.TIMESFORMER.USE_CLS_TOKEN, use_omnivore_vit=cfg.TIMESFORMER.USE_OMNIVORE_VIT, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_small_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=cfg.MODEL.PRETRAINED
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, st_adapter=cfg.TIMESFORMER.ST_ADAPTER, st_adapter_dim=cfg.TIMESFORMER.ST_ADAPTER_DIM, tublet_dim=cfg.TIMESFORMER.TUBLET_DIM, use_time_embed=cfg.TIMESFORMER.USE_TIME_EMBED, use_cls_token=cfg.TIMESFORMER.USE_CLS_TOKEN, use_omnivore_vit=cfg.TIMESFORMER.USE_OMNIVORE_VIT, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
