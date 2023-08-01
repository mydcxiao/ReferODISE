
import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type
from collections import OrderedDict
from .common import LayerNorm2d

import open_clip

from torch.nn import functional as F

from timm.models.layers import trunc_normal_


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        clip_model,
        pretrained,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)


        self.clip = ClipAdapter(clip_model, pretrained=pretrained) if clip_model else None
        
        self.text_dim = self.clip.dim_latent if clip_model else None
        # self.img_size = self.clip.visual.image_size
        # self.text_proj = PositionalLinear(self.text_dim, self.embed_dim, seq_len=77)
        self.text_proj = PositionalLinear(self.text_dim, self.embed_dim, seq_len=1) if clip_model else None

    def forward(self,
                texts: Optional[torch.Tensor],
                ) -> torch.Tensor:
        bs = self._get_batch_size(texts)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if texts is not None:
            texts = texts.squeeze(1)
            text_masks = (texts != 0).long()
            text_embed, text_encodings = self.clip._encode_text(texts)
            text_embed, text_encodings = text_embed.float(), text_encodings.float()
            text_embed = text_embed.unsqueeze(1)
            text_embeddings = self.text_proj(text_embed)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        return sparse_embeddings

    def _get_batch_size(
        self,
        texts: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if texts is not None:
            return texts.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.clip.device


class PositionalLinear(nn.Module):
    def __init__(self, in_features, out_features, seq_len=77, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, out_features))
        self.norm = nn.LayerNorm(out_features)
        trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        x = self.linear(x)
        x = x + self.positional_embedding
        x = self.norm(x)
        return x

class ClipAdapter(nn.Module):
    def __init__(self, 
                 clip_model,
                 pretrained,
                 normalize=True,
        ):

        openai_clip, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        super().__init__()
        self.clip = openai_clip
        # the first two are Resize and Crop, the last one is normalization
        # self.clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])
        self._freeze()
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.normalize = normalize

    def _freeze(self):
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    @property
    def device(self):
        return next(self.parameters()).device

    # don't save clip model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    @property
    def dim_latent(self):
        return self.clip.text_projection.shape[-1]

    @property
    def image_size(self):
        if isinstance(self.clip.visual.image_size, tuple):
            return self.clip.visual.image_size
        else:
            return (self.clip.visual.image_size, self.clip.visual.image_size)

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length
    
    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _encode_text(self, text):
        x = self.clip.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)
        text_encodings = x
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return text_embed, text_encodings