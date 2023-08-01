import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import torchvision
from torchvision.transforms.functional import resize, to_pil_image 
from torchvision.transforms import InterpolationMode

from utils import transforms as T
from utils import utils
import numpy as np
import torch.nn.functional as F
import gc
import torch.distributed as dist
from PIL import Image
import random


def computeIoU(pred, # B x C x H x W 
               gt, # B x H x W
               ):
    gt = gt.unsqueeze(1)
    gt = gt.repeat_interleave(pred.size(1),dim=1)
    I = torch.sum(torch.mul(pred, gt), (3, 2)) # B x C
    U = torch.sum(torch.add(pred, gt), (3, 2)) - I # B x C
    iou = torch.full_like(U, fill_value=0.0)
    mask = (U != 0)
    iou[mask] = I[mask] / U[mask]
    return iou, I, U # B x C

def IoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def generate_random_color(num_masks):
    color_list = []
    while num_masks > 0:
        color = tuple(random.randint(0, 255) for _ in range(3))
        if color not in color_list:
            color_list.append(color)
            num_masks -= 1
    return color_list


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, 
                    epoch, print_freq, iterations, 
                    writer = None,
                    ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    total_loss = 0
    total_its = 0
    
    total_loss_masks = 0
    total_loss_contrastive = 0

    #evaluation variables
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    # torch.autograd.set_detect_anomaly(True)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        
        image, target, sentences, original_size = data
        image, target, sentences = image.cuda(non_blocking=True),\
                                   target.cuda(non_blocking=True),\
                                   sentences.cuda(non_blocking=True),\

        low_res_logits, sim_pred = model(image, sentences)
        mask_slice = slice(0, 1)
        low_res_logits = low_res_logits[:,mask_slice,:,:]
        loss, loss_masks, loss_contrastive = criterion(low_res_logits, target, sim_pred) 

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        total_loss += loss.item()

        total_loss_masks += loss_masks.item()
        total_loss_contrastive += loss_contrastive.item()

        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        # for evaluation
        with torch.no_grad():
            low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)
            iou, I, U = computeIoU(low_res_masks, target) # B x C
            iou, idx = iou.max(1)
            I = I[range(len(idx)), idx]
            U = U[range(len(idx)), idx]

            acc_ious += iou.mean().item()
            mean_IoU.append(iou.mean().item())
            cum_I += I.sum().item()
            cum_U += U.sum().item()

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                for i in range(len(idx)):
                    if iou[i].item() >= eval_seg_iou:
                        seg_correct[n_eval_iou] += 1
            seg_total += len(idx)


        # for summary writer
        if writer is not None:
            writer.add_scalar(f'train_per_iter/{utils.get_rank():d}_loss', loss.item(), 
                               iterations + len(data_loader) * epoch)
            writer.add_scalar('train_per_iter/lr', optimizer.param_groups[0]["lr"], 
                               iterations + len(data_loader) * epoch)

            writer.add_scalar(f'train_per_iter/{utils.get_rank():d}_loss_masks', loss_masks.item(),
                               iterations + len(data_loader) * epoch)
            writer.add_scalar(f'train_per_iter/{utils.get_rank():d}_loss_contrastive', loss_contrastive.item(),
                               iterations + len(data_loader) * epoch)

        del image, target, sentences, data, \
            low_res_logits, sim_pred, loss, loss_masks, loss_contrastive, \
            low_res_masks, iou, I, U


        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # metric_logger.synchronize_between_processes()
    # total_loss = torch.tensor(total_loss, dtype=torch.float64, device='cuda')
    # dist.barrier()
    # dist.all_reduce(total_loss)
    # total_loss = total_loss.item()
    # print("Train loss: {:4f} ({:4f})\n".format(metric_logger.meters['loss'].total, total_loss))
    
    iou = acc_ious / total_its
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total, total_loss, total_loss_masks, total_loss_contrastive], 
                      dtype=torch.float64, device='cuda')
    seg = torch.tensor(seg_correct, dtype=torch.int32, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    dist.all_reduce(seg)
    t = t.tolist()
    iou = t[0] / utils.get_world_size()
    mIoU = t[1] / utils.get_world_size()
    cum_I = t[2]
    cum_U = t[3]
    seg_total = int(t[4])
    total_loss = t[5]
    total_loss_masks = t[6]
    total_loss_contrastive = t[7]
    seg_correct = seg.tolist()

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    print("Train loss: {:4f}\n".format(total_loss))

    # for summary writer
    if writer is not None:
        writer.add_scalar('train_per_epoch/loss', total_loss, epoch)
        writer.add_scalar('train_per_epoch/mIoU', mIoU * 100., epoch)
        writer.add_scalar('train_per_epoch/average_IoU', 100 * iou, epoch)
        writer.add_scalar('train_per_epoch/overall_IoU', cum_I * 100. / cum_U, epoch)

        writer.add_scalar('train_per_epoch/loss_masks', total_loss_masks, epoch)
        writer.add_scalar('train_per_epoch/loss_contrastive', total_loss_contrastive, epoch)



def eval_train(model, data_loader, epoch, criterion,
               writer = None,
               ):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    total_loss = 0
    total_loss_masks = 0
    total_loss_contrastive = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 50, header):
            total_its += 1
            image, target, sentences, original_size = data

            image, target, sentences = image.cuda(non_blocking=True),\
                                       target.cuda(non_blocking=True),\
                                       sentences.cuda(non_blocking=True)

            low_res_logits, sim_pred = model(image, sentences)

            mask_slice = slice(0, 1)
            low_res_logits = low_res_logits[:, mask_slice, :, :]

            loss, loss_masks, loss_contrastive = criterion(low_res_logits, target, sim_pred) 

            total_loss += loss.item()
            total_loss_masks += loss_masks.item()
            total_loss_contrastive += loss_contrastive.item()

            low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)

            iou, I, U = computeIoU(low_res_masks, target) # B x C

            iou, idx = iou.max(1)
            I = I[range(len(idx)), idx]
            U = U[range(len(idx)), idx]

            acc_ious += iou.mean().item()
            mean_IoU.append(iou.mean().item())
            cum_I += I.sum().item()
            cum_U += U.sum().item()

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                for i in range(len(idx)):
                    if iou[i].item() >= eval_seg_iou:
                        seg_correct[n_eval_iou] += 1
            seg_total += len(idx)

            #-----------------------------------------------------------------------------------------------
            # for summary writer
            if writer is not None:
                original_img = image
                img_ndarray = original_img.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                target1 = target.cpu().data.numpy().astype(np.uint8)
                target2 = low_res_masks[range(len(idx)), idx].squeeze(1).cpu().data.numpy()
                target2 = target2.astype(np.uint8)
                for i in range(img_ndarray.shape[0]):
                    img = img_ndarray[i, :, : ,:]
                    img = np.array(resize(to_pil_image(img), tuple(original_size[i].int().tolist())))
                    mask1 = np.array(resize(to_pil_image(target1[i]), tuple(original_size[i].int().tolist()), interpolation=InterpolationMode.NEAREST))
                    mask2 = np.array(resize(to_pil_image(target2[i]), tuple(original_size[i].int().tolist()), interpolation=InterpolationMode.NEAREST))
                    visualization1 = overlay_davis(img, mask1, colors=[[0,0,0],[0,255,0]])
                    visualization2 = overlay_davis(img, mask2)
                    visualization = 0.5 * visualization1 + 0.5 * visualization2
                    visualization = visualization.astype(img.dtype)
                    writer.add_image(f'eval_val/{utils.get_rank():d}_{total_its:d}_{i:d}'
                                     , visualization, epoch, dataformats='HWC')
                    writer.add_text(f'eval_val/{utils.get_rank():d}_{total_its:d}_{i:d}',
                                    f'{loss.item():.4f}', epoch)
            #-----------------------------------------------------------------------------------------------

            torch.cuda.synchronize()

            del image, target, sentences, data, \
                low_res_logits, sim_pred, low_res_masks, iou, I, U, idx, \
                loss, loss_masks, loss_contrastive  
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)

    t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total, total_loss, total_loss_masks, total_loss_contrastive], 
                     dtype=torch.float64, device='cuda')
    # t = torch.tensor([iou, mIoU, cum_I, cum_U, seg_total], dtype=torch.float64, device='cuda')
    seg = torch.tensor(seg_correct, dtype=torch.int32, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    dist.all_reduce(seg)
    t = t.tolist()
    iou = t[0] / utils.get_world_size()
    mIoU = t[1] / utils.get_world_size()
    cum_I = t[2]
    cum_U = t[3]
    seg_total = int(t[4])
    total_loss = t[5]
    total_loss_masks = t[6]
    total_loss_contrastive = t[7]
    seg_correct = seg.tolist()

    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    #------------------------------------------------------------------------------------------------
    # output validation iou in summary writer
    if writer is not None:
        writer.add_scalar('eval_val/global_mIoU', mIoU * 100., epoch)
        writer.add_scalar('eval_val/global_average_IoU', 100 * iou, epoch)
        writer.add_scalar('eval_val/global_overall_IoU', cum_I * 100. / cum_U, epoch)
        writer.add_scalar('eval_val/global_loss', total_loss, epoch)
        writer.add_scalar('eval_val/global_loss_masks', total_loss_masks, epoch)
        writer.add_scalar('eval_val/global_loss_contrastive', total_loss_contrastive, epoch)
    #------------------------------------------------------------------------------------------------

    return 100 * iou, 100 * cum_I / cum_U



def eval_test(model, data_loader, device, writer=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    iterations = 0
    pred_iou = 0
    color_list = generate_random_color(model.mask_decoder.num_multimask_outputs)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, original_size = data
            image, target, sentences = image.to(device), target.to(device), \
                                       sentences.to(device)
            
            iterations += 1

            for j in range(sentences.size(-1)):
                low_res_logits, _ = model(image, sentences[:, :, :, j])
                low_res_masks = torch.where(low_res_logits > 0.0, 1, 0)
                low_res_mask = low_res_masks.clone()[:, 0, :, :]
                low_res_mask = low_res_mask.cpu().data.numpy()
                low_res_masks = low_res_masks.cpu().data.numpy()

                target_ = target.cpu().data.numpy()
                I, U = IoU(low_res_mask, target_)
                this_iou = 0.0 if U == 0 else I*1.0/U
                mean_IoU.append(this_iou)
                #----------------------------------------------------
                # output images in summary writer
                if writer is not None:
                    original_img = image
                    img_ndarray = original_img.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                    target1 = target_.astype(np.uint8)
                    target2 = low_res_mask
                    target2 = target2.astype(np.uint8)
                    all_masks = low_res_masks.astype(np.uint8)
                    for i in range(img_ndarray.shape[0]):
                        img = img_ndarray[i, :, : ,:]
                        img = np.array(resize(to_pil_image(img), tuple(original_size[i].int().tolist())))
                        mask1 = np.array(resize(to_pil_image(target1[i]), tuple(original_size[i].int().tolist()), interpolation=InterpolationMode.NEAREST))
                        mask2 = np.array(resize(to_pil_image(target2[i]), tuple(original_size[i].int().tolist()), interpolation=InterpolationMode.NEAREST))
                        visualization1 = overlay_davis(img, mask1, colors=[[0,0,0],[0,255,0]])
                        visualization2 = overlay_davis(img, mask2)
                        visualization = 0.5 * visualization1 + 0.5 * visualization2
                        visualization = visualization.astype(img.dtype)
                        writer.add_image(f'eval_test/{iterations:d}_{this_iou:.2f}', visualization, iterations,
                                        dataformats='HWC')
                        masks = all_masks[i]
                        mask_list = []
                        for j in range(masks.shape[0]):
                            mask = masks[j,:,:]
                            mask = np.array(resize(to_pil_image(mask), tuple(original_size.int().tolist()), interpolation=InterpolationMode.NEAREST))
                            mask_list.append(mask)
                        masks = np.stack(mask_list, axis=0)
                        vis = overlay_masks(img, masks, colors=color_list)
                        writer.add_image(f'all_masks/{iterations:d}_{this_iou:.2f}', vis, iterations,
                                        dataformats='HWC')
                        
                #----------------------------------------------------
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, data, \
                original_size, input_size, \
                low_res_logits, low_res_masks, low_res_mask, I, U, this_iou

        pred_iou = pred_iou / seg_total

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)



# overlay mask and image for visualization
# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)



def overlay_masks(image, masks, colors, cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    for i in range(masks.shape[0]):
        mask = masks[i,:,:]
        object_ids = np.unique(mask)

        for object_id in object_ids[1:]:
            # Overlay color on  binary mask
            foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[i])
            binary_mask = mask == object_id

            # Compose image
            im_overlay[binary_mask] = foreground[binary_mask]

            # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
            countours = binary_dilation(binary_mask) ^ binary_mask
            # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
            im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)