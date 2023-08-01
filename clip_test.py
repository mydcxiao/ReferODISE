import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

# from .common import LayerNorm2d

import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
print(preprocess)
print(model.visual.image_size)
print(model.text_projection.shape[-1])
# clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])

# tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
# text = tokenizer(["a diagram", "a dog", "a cat"])
# tokenizer = open_clip.tokenize
# text = tokenizer(["a diagram", 'a dog', 'a cat'])
# text = open_clip.tokenize("a diagram")
# text = tokenizer(["a diagram"])
# text = tokenizer("photo of a diagram")
# print(text)
# val, idx = text.max(dim=-1)
# print(text.size())
# print(val)
# print(idx)
# print(model.text_projection.size())
# text_mask = (text != 0).long()
# x = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
# x = x + model.positional_embedding
# x = x.permute(1, 0, 2)  # NLD -> LND
# x = model.transformer(x, attn_mask=model.attn_mask)
# x = model.transformer(x, attn_mask=text_mask)
# x = x.permute(1, 0, 2)  # LND -> NLD
# x = model.ln_final(x)
# text_encodings = x
# text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
# print(text_encodings.size(), text_embed.size())
# print(model.attn_mask.size())


# text = text.unsqueeze(1)
# print(text.size())
# text = text[:, :20]
# for param in model.parameters():
#     # param.requires_grad = False
#     print(param.requires_grad)
# with torch.no_grad(), torch.cuda.amp.autocast():
# #     # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
# #     # image_features /= image_features.norm(dim=-1, keepdim=True)
# #     text_features /= text_features.norm(dim=-1, keepdim=True)
#     # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(text_features)
# print(text_features.size()) # 3 * 768
# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

class MaskCLIP(nn.Module):
    def __init__(
        self,
        clip,
        img_size: int = 336,
    ):
        super().__init__()
        self.clip = clip
        self.clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])
        self.img_size = img_size

    def _mask_clip_forward(self, x: torch.Tensor, attn_mask: torch.Tensor, num_mask_tokens: int):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        x = self.clip.visual.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # [N, L, D]
        x = self.clip.visual.ln_post(x[:, :num_mask_tokens, :])

        if self.clip.visual.proj is not None:
            x = torch.einsum("nld,dc->nlc", x, self.clip.visual.proj)

        return x

    def encode_image_with_mask(self, image, mask):
        assert hasattr(self.clip.visual, "positional_embedding")
        image = self.clip_preprocess(image)
        batch_size = image.shape[0]
        assert batch_size == mask.shape[0]
        num_queries = mask.shape[1]

        # [B, Q, H, W], Q is the number of quries, H and W are the height and width of the image
        mask = mask.sigmoid()
        # [B, Q, H//P, W//P]
        patch_mask = F.max_pool2d(
            mask,
            kernel_size=self.clip.visual.conv1.kernel_size,
            stride=self.clip.visual.conv1.stride,
        )
        # 0 means not masked out, 1 mean masked out
        # so if 1 pixel > 0.5, it is not masked out
        # aka if all pixels (max pixel) < 0.5, it is masked out
        mask_token_attn_mask = patch_mask < 0.5
        # [B, Q, H//P x W//P]
        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)

        num_mask_token = num_queries
        num_image_cls_token = self.clip.visual.positional_embedding.shape[0]
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token

        # we start with no mask out
        attn_mask = torch.zeros(
            (num_all_token, num_all_token), dtype=torch.bool, device=image.device
        )

        # mask+cls+image token to mask token attention is masked out
        attn_mask[:, :num_mask_token] = True

        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.clip.visual.conv1.out_channels // 64  # head width 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * num_heads, num_all_token, num_all_token)

        return self._mask_clip_forward(image, attn_mask, num_mask_token)

    def get_mask_embed(self, image, mask):

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # [B, Q, C]
        mask_embed = self.encode_image_with_mask(image, mask)

        return mask_embed