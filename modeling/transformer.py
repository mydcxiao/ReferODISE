
import torch
from torch import Tensor, nn

import torch.nn.functional as F

import math
from typing import Tuple, Type, List

from .common import MLPBlock, PositionEmbeddingRandom, Mlp, LayerNorm2d

from timm.models.layers import DropPath, drop_path


class MultiScaleTwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        dropout: float = 0.0,
        droppath: float = 0.0,
        sr_ratios: List[int] = [1, 2, 4, 8],
    ) -> None:
        
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        self.dropout = dropout
        self.droppath = droppath

        self.pe_layer = PositionEmbeddingRandom(embedding_dim // 2)

        for i in range(depth):
            self.layers.append(
                MultiScaleTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    dropout=dropout,
                    droppath=droppath,
                    sr_ratio=sr_ratios[i % len(sr_ratios)],
                )
            )
        
        self.linear_s5 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_s4 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_s3 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_s2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, 1, 0),
                            LayerNorm2d(embedding_dim),
                            activation(),
        )

        self.final_attn_token_to_image = ReductionAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embeddings: List[Tensor],
        query_embedding: Tensor,
        num_latent_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
       
        # Prepare queries
        queries = query_embedding

        # Apply transformer blocks and final layernorm
        for i in range(self.depth):
            feat_level = i % len(image_embeddings)
            keys = image_embeddings[feat_level]
            _, _, h, w = keys.shape
            key_pe = self.pe_layer((h, w)).unsqueeze(0)
            queries, keys = self.layers[i](
                queries=queries,
                keys=keys,
                query_pe=query_embedding,
                key_pe=key_pe,
                num_latent_tokens=num_latent_tokens,
            )
            image_embeddings[feat_level] = keys
        s5, s4, s3, s2 = image_embeddings
        s5_ = self.linear_s5(s5.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(s5.shape)
        s5_ = F.interpolate(s5_, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        s4_ = self.linear_s4(s4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(s4.shape)
        s4_ = F.interpolate(s4_, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        s3_ = self.linear_s3(s3.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(s3.shape)
        s3_ = F.interpolate(s3_, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        s2_ = self.linear_s2(s2.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(s2.shape)
        c = self.linear_fuse(torch.cat([s5_, s4_, s3_, s2_], dim=1).contiguous())
        keys = F.dropout2d(c, p=self.dropout, training=self.training)

        # Apply the final attention layer from the points to the image
        _, _, h, w = keys.shape
        key_pe = self.pe_layer((h, w)).unsqueeze(0)
        q = queries + query_embedding
        # k = keys + key_pe
        attn_out, _ = self.final_attn_token_to_image(q=q, img=keys, img_pe=key_pe)
        # queries = queries + attn_out
        queries = drop_path(attn_out, self.droppath, self.training) + queries
        queries = self.norm_final_attn(queries)

        return queries, keys


class MultiScaleTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        dropout: float = 0.0,
        droppath: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = LayerNorm2d(embedding_dim)

        self.two_way_cross_attn = ReductionAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, dropout=dropout, sr_ratio=sr_ratio,
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = LayerNorm2d(embedding_dim)

        self.mlp_q = MLPBlock(embedding_dim, mlp_dim, activation, dropout=dropout)
        self.norm5 = nn.LayerNorm(embedding_dim)
        
        self.mlp_img = Mlp(embedding_dim, mlp_dim, embedding_dim, act_layer=activation, drop=dropout)
        self.norm6 = LayerNorm2d(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

        self.dropout = dropout
        self.droppath = droppath

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor,
        num_latent_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = drop_path(attn_out, self.droppath, self.training) + queries
        queries = self.norm1(queries)
        
        keys = self.norm2(keys)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        # k = keys + key_pe
        q_out, img_out = self.two_way_cross_attn(q=q, img=keys, img_pe=key_pe)
        q_out = self.norm3(q_out)
        img_out = self.norm4(img_out)

        # MLP block
        mlp_out_q = self.mlp_q(q_out)
        queries = drop_path(mlp_out_q, self.droppath, self.training) + queries
        queries = self.norm5(queries)

        B, C, H, W = img_out.shape
        img_out = img_out.flatten(2).permute(0, 2, 1)
        mlp_out_img = self.mlp_img(img_out, H, W)
        keys = drop_path(mlp_out_img, self.droppath, self.training).transpose(1, 2).reshape(B, C, H, W) + keys
        keys = self.norm6(keys)

        return queries, keys

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        
        # Dropout
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class ReductionAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        
        self.dropout = dropout
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(self.embedding_dim)
            self.kv = nn.Linear(embedding_dim, embedding_dim * 2)
            self.si = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=sr_ratio, stride=sr_ratio)
        
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, img: Tensor, img_pe: Tensor) -> Tensor:

        B, C, H, W = img.shape

        if self.sr_ratio > 1:
            img_ = self.sr(img).reshape(B, C, -1).permute(0, 2, 1)
            img_ = self.norm(img_)
            img_pe_ = self.sr(img_pe).reshape(1, C, -1).permute(0, 2, 1)
            kv = self.kv(img_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
            k, v = kv[0] + img_pe_, kv[1]
        else:
            img_ = img.reshape(B, C, -1).permute(0, 2, 1)
            img_pe_ = img_pe.reshape(1, C, -1).permute(0, 2, 1)
            k, v = img_ + img_pe_, img_

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x C_per_head x N_tokens
        
        attn_T = attn.permute(0, 1, 3, 2)

        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        attn_T = attn_T / math.sqrt(c_per_head)
        attn_T = torch.softmax(attn_T, dim=-1)

        
        # Dropout
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_T = F.dropout(attn_T, p=self.dropout, training=self.training)

        # Get output
        q_out = attn @ v
        q_out = self._recombine_heads(q_out)
        q_out = self.out_proj(q_out)

        img_out = attn_T @ q
        img_out = self._recombine_heads(img_out)
        img_out = self.out_proj(img_out)
        img_out = img_out.permute(0, 2, 1).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
        if self.sr_ratio > 1:
            img_out = self.si(img_out)

        return q_out, img_out


class GumbelAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout = dropout
        self.tau = nn.Parameter(torch.ones([]) * tau)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, n: int) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        q1 = q[:, :n]
        q2 = q[:, n:]
        q1 = q1 / q1.norm(dim=-1, keepdim=True)
        k1 = k / k.norm(dim=-1, keepdim=True)

        # Separate into heads
        q1 = self._separate_heads(q1, self.num_heads)
        q2 = self._separate_heads(q2, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        k1 = self._separate_heads(k1, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q1.shape
        attn1 = q1 @ k1.permute(0, 1, 3, 2)  # B x N_heads x C_per_head x N_tokens
        attn2 = q2 @ k.permute(0, 1, 3, 2)  # B x N_heads x C_per_head x N_tokens
        attn2 = attn2 / math.sqrt(c_per_head)
        attn2 = torch.softmax(attn2, dim=-1)
        attn1 = F.gumbel_softmax(attn1, tau=self.tau, hard=True, dim=-2)
        
        # Dropout
        attn1 = F.dropout(attn1, p=self.dropout, training=self.training)
        attn2 = F.dropout(attn2, p=self.dropout, training=self.training)

        # Get output
        out1 = attn1 @ v
        out2 = attn2 @ v
        out = torch.cat([out1, out2], dim=2)
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out