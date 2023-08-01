import torch
import torch.nn as nn

from modeling import M, FeatureExtractorBackbone, LdmImplicitCaptionerExtractor, MaskDecoder, PromptEncoder, MultiScaleTwoWayTransformer, MSDeformAttnPixelDecoder
from detectron2.modeling import ShapeSpec


def build_m_ldm(ck=None,
                ck_image_encoder=None,
                ck_pixel_decoder=None,
                ck_prompt_encoder=None,
                ck_mask_decoder=None):
    m = _build_m(
        ck=ck,
        ck_image_encoder=ck_image_encoder,
        ck_pixel_decoder=ck_pixel_decoder,
        ck_prompt_encoder=ck_prompt_encoder,
        ck_mask_decoder=ck_mask_decoder,
        is_ldm=True,
    )
    m.hook_clip_model()
    return m


m_model_registry = {
    "default": build_m_ldm,
    "ldm": build_m_ldm,
}


def _build_m(
    ck=None,
    ck_image_encoder=None,
    ck_pixel_decoder=None,
    ck_prompt_encoder=None,
    ck_mask_decoder=None,
    is_ldm=False,
):
    prompt_embed_dim = 256
    image_embedding_size = 64
    out_feature_strides = {
            "s2": 4,
            "s3": 8,
            "s4": 16,
            "s5": 32,
    }
    out_feature_channels = {
            "s2": 512,
            "s3": 512,
            "s4": 512,
            "s5": 512,
    }
    out_features=["s2", "s3", "s4", "s5"]
    input_shape = {
            name: ShapeSpec(
                channels=out_feature_channels[name], stride=out_feature_strides[name]
            )
            for name in out_features
    }
    clip_model = None if is_ldm else 'ViT-L-14-336'
    pretrained = None if is_ldm else 'openai'
    m = M(
        image_encoder = FeatureExtractorBackbone(
            feature_extractor=LdmImplicitCaptionerExtractor(
                encoder_block_indices=(5, 7),
                unet_block_indices=(2, 5, 8, 11),
                decoder_block_indices=(2, 5),
                steps=(0,),
                learnable_time_embed=True,
                num_timesteps=1,
                clip_model_name="ViT-L-14-336",
            ),
            out_features=out_features,
            use_checkpoint=True,
            slide_training=True,
        ),
        pixel_decoder=MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            transformer_in_features=["s3", "s4", "s5"],
            common_stride=4,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            clip_model=clip_model,
            pretrained=pretrained,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=100,
            transformer=MultiScaleTwoWayTransformer(
                depth=8,
                embedding_dim=prompt_embed_dim,
                mlp_dim=1024,
                activation=nn.GELU,
                num_heads=8,
                dropout=0.1,
                droppath=0.4,
                # sr_ratios = [1, 2, 4, 8],
                sr_ratios = [1, 1, 1, 1],
            ),
            transformer_dim=prompt_embed_dim,
        ),
    )
    if ck:
        with open(ck, "rb") as f:
            state_dict = torch.load(f)
        m.load_state_dict(state_dict['model'], strict=False)
    else:
        if ck_image_encoder:
            with open(ck_image_encoder, "rb") as f:
                state_dict = torch.load(f)
            m.image_encoder.load_state_dict(state_dict, 
                                            strict=False
                                           )
        if ck_pixel_decoder:
            with open(ck_pixel_decoder, "rb") as f:
                state_dict = torch.load(f)
            m.pixel_decoder.load_state_dict(state_dict)
        if ck_prompt_encoder:
            with open(ck_prompt_encoder, "rb") as f:
                state_dict = torch.load(f)
            m.prompt_encoder.load_state_dict(state_dict, strict=False)
        if ck_mask_decoder:
            with open(ck_mask_decoder, "rb") as f:
                state_dict = torch.load(f)
            m.mask_decoder.load_state_dict(state_dict, strict=False)
    return m

# model = _build_m(ck_image_encoder='./pretrained/image_encoder/ldm_encoder/caption_backbone.pth', 
#                  ck_pixel_decoder='./pretrained/pixel_decoder/caption_pixel_decoder.pth')

# device = torch.device('cuda:2')  # Specify the device string

# model = model.to(device)

# import open_clip
# tokenizer = open_clip.tokenize
# input_ids = tokenizer(['raw'] * 4)

# batched_images = torch.randn(4,3,1024,1024).to(device)
# batched_sents = input_ids.unsqueeze(1).to(device)

# mask_features, transformer_encoder_features, multi_scale_features = model(batched_images, batched_sents)

# print(batched_outputs[0].shape)
# print(batched_outputs[1].shape)

# print(batched_outputs['s2'].shape) # //4
# print(batched_outputs['s3'].shape) # //8
# print(batched_outputs['s4'].shape) # //16
# print(batched_outputs['s5'].shape) # //32

# print(mask_features.shape)
# print(transformer_encoder_features.shape)
# print(type(multi_scale_features))
# print(multi_scale_features[0].shape)
# print(multi_scale_features[1].shape)
# print(multi_scale_features[2].shape)

# low_res_masks, sim_pred = model(batched_images, batched_sents)

# print(low_res_masks.shape)
# print(sim_pred.shape)