# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from models.transformer.transformer_util import get_2d_sincos_pos_embed_flexible
from functools import partial



class AutoEncoderViT(nn.Module):
    def __init__(self, input_size: int, num_patches: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int, initialize_weights=True, use_cls=True):
        super().__init__()
        self.patch_embed = nn.Linear(input_size, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.embed_dim = encoder_embed_dim
        self.encoder_depths = encoder_depths
        self.mlp_ratio = 4.

        self.input_size = (num_patches, encoder_embed_dim)
        self.patch_size = (1, encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w

        # MAE Encoder
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)
        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # MAE Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, input_size, bias=True)
        self.use_cls = use_cls
        if initialize_weights:
            self.initialize_weights()

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        return latent, pred

    def forward_encoder(self, x: torch.Tensor):
        # embed patches
        #print(f"x shape {x.shape}")
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        if self.use_cls:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        #print(f"x shape {x.shape}")
        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        return x

    def forward_decoder(self, x):
        # embed tokens
        if self.use_cls:
            x = x[:, 1:, :]
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_block:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     (self.grid_h, self.grid_w),
                                                     cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1],
                                                             (self.grid_h, self.grid_w),
                                                             cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)