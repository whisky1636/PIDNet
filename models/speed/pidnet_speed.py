# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model_utils_speed import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag,CoordAtt,PAPPM_optimized
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

def parse_args():
    parser = argparse.ArgumentParser(description='Speed Measurement')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='number of classes', type=int, default=19)
    parser.add_argument('--r', help='input resolution', type=int, nargs='+', default=(1024,2048))     


    args = parser.parse_args()

    return args


class MDFF_Up(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super(MDFF_Up, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        # 不使用 3x3 卷积去平滑混叠效应，直接使用 1x1 卷积进行通道融合/降维
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_low, x_high):
        # 对深层特征进行双线性插值上采样，恢复到浅层特征的分辨率
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=algc)

        # 分别进行1x1卷积调整通道
        x_low_proj = self.conv_low(x_low)
        x_high_proj = self.conv_high(x_high_up)

        # 将浅层特征与上采样后的深层特征拼接，并经过 1x1 卷积融合输出
        out = torch.cat([x_low_proj, x_high_proj], dim=1)
        out = self.conv_fuse(out)
        return out


class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # ---------------------------------------------------------
        # 新增: I Branch 的 MDFF-Up 自顶向下融合模块
        # 注: SPP 模块将先作用于最深层特征输出 planes * 4，然后再自顶向下融合
        # ---------------------------------------------------------
        self.mdff_5_4 = MDFF_Up(planes * 8, planes * 4, planes * 8)  # i_x4(planes*8) & spp_out(planes*4)
        self.mdff_4_3 = MDFF_Up(planes * 4, planes * 8, planes * 4)  # i_x3(planes*4) & fused_4(planes*8)
        self.mdff_3_2 = MDFF_Up(planes * 2, planes * 4, planes * 4)  # i_x2(planes*2) & fused_3(planes*4)，最终输出 planes*4

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM_optimized(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)

        # ---------------------------------------------------------
        # 1. 计算 I Branch 并原封不动地向 P/D 分支传递原始特征
        # ---------------------------------------------------------
        i_x2 = self.relu(self.layer2(self.relu(x)))

        # P / D 初始化使用 I 分支的原始第二层特征
        x_ = self.layer3_(i_x2)
        x_d = self.layer3_d(i_x2)

        i_x3 = self.relu(self.layer3(i_x2))

        # P / D 融合连接使用 I 分支的原始第三层特征 (i_x3)
        x_ = self.pag3(x_, self.compression3(i_x3))
        x_d = x_d + F.interpolate(
            self.diff3(i_x3),
            size=x_.shape[-2:],
            mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_

        i_x4 = self.relu(self.layer4(i_x3))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        # P / D 融合连接使用 I 分支的原始第四层特征 (i_x4)
        x_ = self.pag4(x_, self.compression4(i_x4))
        x_d = x_d + F.interpolate(
            self.diff4(i_x4),
            size=x_.shape[-2:],
            # size=[height_output, width_output],  # size=x_.shape[-2:],
            mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d

        i_x5 = self.layer5(i_x4)
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))

        # 最深层经过 SPP 提取上下文 (提取出的特征维度为 planes * 4)
        i_x5_spp = self.spp(i_x5)

        # ---------------------------------------------------------
        # 2. I 分支使用 MDFF-Up 自顶向下融合特征作为最终表征
        #    (此时分辨率逐步恢复，无需一次性做极端的双线性插值)
        # ---------------------------------------------------------
        fused_i4 = self.mdff_5_4(i_x4, i_x5_spp)  # 融合到 1/32
        fused_i3 = self.mdff_4_3(i_x3, fused_i4)  # 融合到 1/16
        fused_i2 = self.mdff_3_2(i_x2, fused_i3)  # 融合到 1/8

        # 最终输入到 DFM 的 I分支 特征即为 MDFF_Up 融合后的结果
        x_i = fused_i2

        # DFM 接收经过 P分支 (x_)、增强 I分支 (x_i) 和 D分支 (x_d) 的特征融合
        x_ = self.final_layer(self.dfm(x_, x_i, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def get_seg_model(cfg, imgnet_pretrained):
    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128,
                       augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128,
                       augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256,
                       augment=True)

    if imgnet_pretrained:
        # 加载权重文件中的 state_dict (模型参数字典)
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict']
        model_dict = model.state_dict()  # 获取当前初始化的模型参数结构

        # 【核心对齐逻辑】：过滤预训练权重
        # 只有当预训练权重的 Key 在当前模型中存在，且 Tensor 的形状(shape)完全一致时才保留
        # 这可以防止因类别数不同（如 ImageNet 是 1000 类，而分割任务是 19 类）导致输出层报错
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}

        model_dict.update(pretrained_state)  # 用匹配成功的权重更新当前模型字典
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info(msg)
        model.load_state_dict(model_dict, strict=False)  # 非严格加载，允许部分层不匹配
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()

        # 【切片处理】：k[6:] 逻辑
        # 有时权重是在 DataParallel 模式下保存的，Key 会带有 'module.' 前缀
        # k[6:] 的作用是跳过前 6 个字符（通常是去掉 'model.' 或 'module.' 前缀）来尝试匹配
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                           (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}

        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info(msg)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model


def get_pred_model(name, num_classes):
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model

if __name__ == '__main__':
    args = parse_args()
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name=args.a, num_classes=args.c)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, args.r[0], args.r[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print("fps:",FPS)
    
    
    


