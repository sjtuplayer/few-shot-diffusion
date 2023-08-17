from abc import abstractmethod

import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
import unet as ori_unet
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class style_condtioner(nn.Module):
    def __init__(self):
        super().__init__()
        c_hidden_dim = 512
        self.c_style_time = nn.Sequential(
            ori_unet.SinusoidalPosEmb(64),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 512))
        pos_embedding = ori_unet.SinusoidalPosEmb(512)
        self.c_pos_embedding = pos_embedding(torch.arange(0, 11).cuda())
        self.c_transformer = nn.Transformer(c_hidden_dim, 8, 3, 3)
        self.c_query_pos = nn.Parameter(torch.rand(1, c_hidden_dim)).unsqueeze(1).cuda()
        self.c_factor_linear = nn.Linear(c_hidden_dim, 10)
        self.c_transformer2 = nn.Transformer(c_hidden_dim, 8, 3, 3)
        self.c_query_pos2 = nn.Parameter(torch.rand(10, c_hidden_dim)).unsqueeze(1).cuda()
        self.c_feature_linear = nn.Linear(c_hidden_dim, 512)
        self.c_ta_linear = nn.Linear(512, 512)
        self.c_tb_linear = nn.Linear(512, 512)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x,timesteps,condition):
        b = x.size(0)
        c_t = self.c_style_time(timesteps)
        style_feature = condition['style']
        # print(style_feature.device,self.c_pos_embedding.device,c_t.device,style_feature.device,self.c_query_pos.device)
        # print(c_t.unsqueeze(0).shape,style_feature.unsqueeze(1).repeat(1,b,1).shape,self.c_query_pos.repeat(1, b, 1).shape,self.c_pos_embedding.shape)
        hidden_state = self.c_transformer(self.c_pos_embedding.unsqueeze(1) + torch.cat(
            [c_t.unsqueeze(0), style_feature.unsqueeze(1).repeat(1, b, 1)], dim=0),
                                          self.c_query_pos.repeat(1, b, 1))
        factor = self.c_factor_linear(hidden_state.permute(1, 0, 2)).squeeze()
        factor = self.softmax(factor).unsqueeze(-1)
        hidden_state2 = self.c_transformer2(
            self.c_pos_embedding.unsqueeze(1) + torch.cat(
                [c_t.unsqueeze(0), style_feature.unsqueeze(1).repeat(1, b, 1)], dim=0),
            self.c_query_pos2.repeat(1, b, 1))
        c_feature = self.c_feature_linear(hidden_state2.permute(1, 0, 2))
        merge_feature = (factor * c_feature).sum(dim=1)
        ta = self.c_ta_linear(merge_feature).unsqueeze(-1).unsqueeze(-1)
        tb = self.c_tb_linear(merge_feature).unsqueeze(-1).unsqueeze(-1)
        return ta,tb
class MyUnet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        condition=False
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.condition=condition
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        #print(channel_mult)
        if self.condition:
            # 256*32*32 content
            self.c_content_condition = nn.ModuleList([
                nn.Sequential(
                    ori_unet.SinusoidalPosEmb(64),
                    nn.Linear(64, 256),
                    nn.GELU(),
                    nn.Linear(256, 256)),
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, 2, 1),
                    nn.ReLU()),
                ori_unet.ResnetBlock(256, 256, time_emb_dim=256),
                ori_unet.ResnetBlock(256, 256, time_emb_dim=256),
                ori_unet.Residual(ori_unet.PreNorm(256, ori_unet.LinearAttention(256)))]
            )
            self.c_style_condition=style_condtioner()
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks): #都是1

                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, condition=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """


        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for index,module in enumerate(self.input_blocks):
            h = module(h, emb)
            #print(h.dtype,x.dtype)
            hs.append(h)
            if self.condition and condition is not None and index==7:
                c_time_mlp, c_conv, c_block1, c_block2, c_attn = self.c_content_condition
                t2 = c_time_mlp(timesteps)
                x_content = c_conv(condition['content'])
                x_content = c_block1(x_content, t2)
                x_content = c_block2(x_content, t2)
                x_content = c_attn(x_content)
                h = h + x_content
        h = self.middle_block(h, emb)
        if self.condition and condition is not None:
            ta,tb=self.c_style_condition(x,timesteps,condition)
            h=h*ta+tb
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        h=self.out(h)
        mean, var = th.split(h, 3, dim=1)
        return mean


class SuperResModel(MyUnet):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


# class EncoderUNetModel(nn.Module):
#     """
#     The half UNet model with attention and timestep embedding.
#
#     For usage, see UNet.
#     """
#
#     def __init__(
#         self,
#         image_size,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=1,
#         num_head_channels=-1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         resblock_updown=False,
#         use_new_attention_order=False,
#         pool="adaptive",
#         condition=False
#     ):
#         super().__init__()
#
#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads
#         self.condition=condition
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#
#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )
#
#         ch = int(channel_mult[0] * model_channels)
#         self.input_blocks = nn.ModuleList(
#             [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
#         )
#         self._feature_size = ch
#         input_block_chans = [ch]
#         ds = 1
#         # if self.condition:
#         #     # 256*32*32 content
#         #     self.c_content_condition = nn.ModuleList([
#         #         nn.Sequential(
#         #             ori_unet.SinusoidalPosEmb(64),
#         #             nn.Linear(64, 256),
#         #             nn.GELU(),
#         #             nn.Linear(256, 256)),
#         #         nn.Sequential(
#         #             nn.Conv2d(256, 256, 3, 2, 1),
#         #             nn.ReLU()),
#         #         ori_unet.ResnetBlock(256, 256, time_emb_dim=256),
#         #         ori_unet.ResnetBlock(256, 256, time_emb_dim=256),
#         #         ori_unet.Residual(ori_unet.PreNorm(256, ori_unet.LinearAttention(256)))]
#         #     )
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=int(mult * model_channels),
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 if ds in attention_resolutions:
#                     layers.append(
#                         AttentionBlock(
#                             ch,
#                             use_checkpoint=use_checkpoint,
#                             num_heads=num_heads,
#                             num_head_channels=num_head_channels,
#                             use_new_attention_order=use_new_attention_order,
#                         )
#                     )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch
#
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=num_head_channels,
#                 use_new_attention_order=use_new_attention_order,
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self._feature_size += ch
#         self.pool = pool
#         if pool == "adaptive":
#             self.out = nn.Sequential(
#                 normalization(ch),
#                 nn.SiLU(),
#                 nn.AdaptiveAvgPool2d((1, 1)),
#                 zero_module(conv_nd(dims, ch, out_channels, 1)),
#                 nn.Flatten(),
#             )
#         elif pool == "attention":
#             assert num_head_channels != -1
#             self.out = nn.Sequential(
#                 normalization(ch),
#                 nn.SiLU(),
#                 AttentionPool2d(
#                     (image_size // ds), ch, num_head_channels, out_channels
#                 ),
#             )
#         elif pool == "spatial":
#             self.out = nn.Sequential(
#                 nn.Linear(self._feature_size, 2048),
#                 nn.ReLU(),
#                 nn.Linear(2048, self.out_channels),
#             )
#         elif pool == "spatial_v2":
#             self.out = nn.Sequential(
#                 nn.Linear(self._feature_size, 2048),
#                 normalization(2048),
#                 nn.SiLU(),
#                 nn.Linear(2048, self.out_channels),
#             )
#         else:
#             raise NotImplementedError(f"Unexpected {pool} pooling")
#
#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#
#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#
#     def forward(self, x, timesteps):
#         """
#         Apply the model to an input batch.
#
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :return: an [N x K] Tensor of outputs.
#         """
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
#
#         results = []
#         h = x.type(self.dtype)
#         for module in self.input_blocks:
#             h = module(h, emb)
#             if self.pool.startswith("spatial"):
#                 results.append(h.type(x.dtype).mean(dim=(2, 3)))
#         h = self.middle_block(h, emb)
#         if self.pool.startswith("spatial"):
#             results.append(h.type(x.dtype).mean(dim=(2, 3)))
#             h = th.cat(results, axis=-1)
#             return self.out(h)
#         else:
#             h = h.type(x.dtype)
#             return self.out(h)
def create_model(
    image_size,
    num_channels=128,
    num_res_blocks=1,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
        condition=False
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return MyUnet(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(1000 if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        condition=condition
    )

