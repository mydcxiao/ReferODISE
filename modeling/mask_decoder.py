# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim, 3)

        self.similarity_head = MLP(transformer_dim, transformer_dim, transformer_dim, 3)

    def forward(
        self,
        image_embeddings: List[torch.Tensor],
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, sims = self.predict_masks(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return masks, sims

    def predict_masks(
        self,
        image_embeddings: List[torch.Tensor], # B x C x H x W
        sparse_prompt_embeddings: torch.Tensor, # B x 1 x S 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings
        # b, c, h, w = src[-1].shape

        # Run the transformer
        hs, src = self.transformer(src, tokens, self.num_mask_tokens)
        mask_tokens_out = hs[:, : self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src

        hyper_in_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps(mask_tokens_out[:, i, :]))
            logits_list.append(self.similarity_head(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        logits = torch.stack(logits_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        b, l, c = sparse_prompt_embeddings.shape
        sims = (logits @ sparse_prompt_embeddings.transpose(1, 2)).view(b, -1, l).mean(dim=-1)

        # Generate mask quality predictions

        return masks, sims


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
