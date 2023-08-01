import torch
import torch.nn as nn

from typing import Any, Dict, List, Tuple, Optional

from .ldm_encoder.meta_arch.ldm import LdmImplicitCaptionerExtractor
from .ldm_encoder.backbone.feature_extractor import FeatureExtractorBackbone
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder, PositionalLinear

class M(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: FeatureExtractorBackbone,
        pixel_decoder: MSDeformAttnPixelDecoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.pixel_decoder = pixel_decoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        
        self._freeze_image_encoder()
        # self._freeze_pixel_decoder()

    # @property
    # def device(self) -> Any:
    #     return self.pixel_mean.device

    def forward(
        self,
        batched_images, # B x C x H x W
        batched_sents, # B x 1 x D
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings = self.image_encoder(batched_images)
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(image_embeddings)
        multi_scale_features.append(mask_features)
        image_embeddings = multi_scale_features
        sparse_embeddings = self.prompt_encoder(
            texts=batched_sents,
        )
        low_res_masks, sim_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
        )
        
        return low_res_masks, sim_pred
    
    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_image_encoder()
        # self._freeze_pixel_decoder()
        return self

    def _freeze_image_encoder(self):
        self.image_encoder.eval()
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def _freeze_pixel_decoder(self):
        self.pixel_decoder.eval()
        for p in self.pixel_decoder.parameters():
            p.requires_grad = False

    def hook_clip_model(self):
        self.prompt_encoder.clip = self.image_encoder.feature_extractor.clip
        self.prompt_encoder.text_dim = self.prompt_encoder.clip.dim_latent
        self.prompt_encoder.text_proj = PositionalLinear(self.prompt_encoder.text_dim, self.prompt_encoder.embed_dim, seq_len=1)