# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

@torch.no_grad()
def get_confusion_matrix_gpu(label, pred, num_class, ignore=-1):
    # 1. 确保取到最大概率的类别索引
    seg_pred = torch.argmax(pred, dim=1)
    seg_gt = label

    # 2. 严格过滤忽略标签
    mask = (seg_gt >= 0) & (seg_gt < num_class) & (seg_gt != ignore)
    seg_gt = seg_gt[mask].long()
    seg_pred = seg_pred[mask].long()

    # 3. 计算一维索引用于 bincount
    # 必须转为 long 以防止计算溢出
    index = seg_gt * num_class + seg_pred

    # 4. 统计频次
    count = torch.bincount(index, minlength=num_class ** 2)
    return count.reshape(num_class, num_class).float()



def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS

    # 修改1: 初始化混淆矩阵在 GPU 上
    confusion_matrix = torch.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums)).cuda()

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()

            # 修改2: 使用 non_blocking=True 加速数据传输
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            bd_gts = bd_gts.float().cuda(non_blocking=True)

            # 修改3: 开启 AMP 混合精度
            with torch.cuda.amp.autocast():
                losses, pred, _, _ = model(image, label, bd_gts)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            for i, x in enumerate(pred):
                # 插值操作在 GPU 上非常快，保持在 GPU
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                # 修改4: 使用 GPU 版本计算混淆矩阵，直接累加到 GPU Tensor 上
                # 避免了原代码中 .cpu().numpy() 带来的巨大同步开销
                confusion_matrix[..., i] += get_confusion_matrix_gpu(
                    label,
                    x,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            # 这里的 loss.item() 会触发同步，但对于 loss 标量影响较小
            loss = losses.mean()
            ave_loss.update(loss.item())

            # 减少打印频率
            if idx % 50 == 0:
                logging.info(f'Validate iter: {idx}')

    # 修改5: 循环结束后，统一将结果转回 CPU 进行指标计算
    confusion_matrix = confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        logging.info('Output {} Mean IoU: {:.4f}'.format(i, mean_IoU))
        logging.info('Class IoU: {}'.format(IoU_array))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    num_classes = config.DATASET.NUM_CLASSES
    # 在 GPU 上初始化混淆矩阵
    confusion_matrix_gpu = torch.zeros((num_classes, num_classes)).cuda()

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()

            # 开启混合精度推理加速
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            # 保持在 GPU 进行插值，对齐尺寸
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            # 累加 GPU 混淆矩阵 (核心加速点)
            confusion_matrix_gpu += get_confusion_matrix_gpu(
                label.cuda(non_blocking=True),
                pred,
                num_classes,
                config.TRAIN.IGNORE_LABEL
            )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            # 实时显示 mIoU (降低同步频率，每 100 step 才拉取一次数据)
            if index % 100 == 0 and index > 0:
                cm_cpu = confusion_matrix_gpu.cpu().numpy()
                pos = cm_cpu.sum(1)
                res = cm_cpu.sum(0)
                tp = np.diag(cm_cpu)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                logging.info('processing: %d images, live mIoU: %.4f' % (index, IoU_array.mean()))

    # 推理结束，一次性转回 CPU 计算最终指标
    confusion_matrix = confusion_matrix_gpu.cpu().numpy()

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_acc = tp.sum() / np.maximum(1.0, pos.sum())
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc





def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                # 显式将 size 转换为整数元组
                target_size = (int(size[0]), int(size[1]))
                pred = F.interpolate(
                    pred, size=target_size,
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
