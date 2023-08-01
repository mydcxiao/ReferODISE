import torch

state_dict = torch.load('./checkpoints/odise_caption_coco_50e-853cc971.pth')
# state_dict = torch.load('./checkpoints/odise_label_coco_50e-b67d2efc.pth')

for name, weight in state_dict['model'].items():
    print(name)
