# Diffusion Encoder + Pixeldecoder + Mutiscaledecoder
The official PyTorch implementation of diffusion model as an image encoder, combined with Mask2Former's pixel decoder and a self-developed multi-scale decoder.

## Framework

Decoder (architecture refer to [Segformer](https://github.com/NVlabs/SegFormer/tree/master) and [SAM](https://github.com/facebookresearch/segment-anything)
<p align="center">
  <img src="./framework.png" width="1000">
</p>

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org)
   - Stable Diffusion dependencies(https://github.com/CompVis/stable-diffusion)
2. Datasets
   - The detailed instruction is in [LAVT](https://github.com/yz93/LAVT-RIS).
3. Pretrained weights
   - refer to [ODISE](https://github.com/NVlabs/ODISE)

## Train and Test

Refer to [LAVT](https://github.com/yz93/LAVT-RIS).

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.


Some code changes come from [CRIS](https://github.com/DerrickWang005/CRIS.pytorch/tree/master) and [LAVT](https://github.com/yz93/LAVT-RIS).
