import torch
from torch import nn
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self,
                 num_masks, # B
                 coef_focal, # weight for focal loss
                 coef_dice, # weight for dice loss
                 ):
        super().__init__()
        self.num_masks = num_masks
        self.coef_focal = coef_focal
        self.coef_dice = coef_dice
        self.tau = nn.Parameter(torch.ones([]) * 1.0)


    def loss_masks(self, 
                   src_masks, 
                   target_masks, 
                   ):
        """Compute the losses related to the masks: the focal loss and the dice loss.
            src_masks: b, 3, h, w
            targets masks: b, h, w
            num_masks: b
        """
        # upsample predictions to the target size
        num_multimask = src_masks.size(1)
        target_masks = target_masks.flatten(1)
        
        alpha = -1
        # count_pos = target_masks.sum(1)
        # count_neg = (1 - target_masks).sum(1)
        # alpha = torch.where(count_pos > 0, count_neg / (count_pos + count_neg), 0)
        # alpha = alpha.unsqueeze(1)
        
        # wf = self.coef_focal / (self.coef_focal + self.coef_dice)
        # wd = 1 - wf
        wf = self.coef_focal
        wd = self.coef_dice

        loss_masks_list = []
        for i in range(num_multimask):
            src_mask = src_masks[:, i, :, :].flatten(1)
            # target_masks = target_masks.view(src_masks.shape)
            loss_masks_list.append(wf * self.sigmoid_focal_loss(src_mask, target_masks, alpha) \
                                  + wd * self.dice_loss(src_mask, target_masks))
        loss_masks_tensor = torch.stack(loss_masks_list, dim=1)
        loss_masks_min, _ = loss_masks_tensor.min(1)
        loss_masks = loss_masks_min.sum() / self.num_masks
        return loss_masks


    def dice_loss(self,
                  inputs, 
                  targets, 
                  ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                        classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        # inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss # (B,)


    def sigmoid_focal_loss(self,
                           inputs, 
                           targets, 
                           alpha = -1, 
                           gamma: float = 0, 
                           ):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if isinstance(alpha, torch.Tensor) or alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1) # (B,)

    def contrastive_loss(self,
                         sims, # B x Q
        ):
        sims = sims / self.tau
        target = torch.zeros_like(sims)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(sims, target)
        return loss

    def matching_loss(self,
                      sims, # B x Q
        ):
        sims_1, _ = sims[:, 1:].max(dim=-1)
        sims = torch.stack([sims[:, 0], sims_1], dim=-1)
        target = torch.zeros_like(sims)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(sims, target)
        return loss
    
    def forward(self, src_masks, target_masks, sims):
        loss_masks = self.loss_masks(src_masks, target_masks)
        loss_contrastive = self.contrastive_loss(sims)
        # loss_matching = self.matching_loss(sims)
        # return loss_masks + loss_contrastive + loss_matching
        # return loss_masks + loss_contrastive
        return loss_masks + loss_contrastive, loss_masks, loss_contrastive

    


