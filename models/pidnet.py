# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 = nn.Sequential(
            # 第一层：3x3 卷积，步长为 2，将图像尺寸缩小为 1/2，通道数变为 planes
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),

            # 第二层：3x3 卷积，步长为 2，图像尺寸再次缩小一半（总共缩小为 1/4），通道数保持 planes
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        # layer1：保持 1/4 分辨率，包含 m 个 BasicBlock，通道数为 planes
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)

        # layer2：步长为 2，将分辨率缩小为 1/8，通道数加倍至 planes * 2
        # 此时的输出 x 会被 P 分支和 D 分支作为初始输入
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        # layer3：步长为 2，分辨率缩小为 1/16，通道数变为 planes * 4
        # 包含 n 个 BasicBlock（通常 n > m）
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)

        # layer4：步长为 2，分辨率缩小为 1/32，通道数变为 planes * 8
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        # layer5：步长为 2，分辨率缩小为 1/64，通道数保持 planes * 8
        # 使用 Bottleneck 结构，expansion=2，内部会先压缩再扩张通道
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        # compression3：将 I 分支 layer3 的输出（planes * 4）压缩到 planes * 2
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        # compression4：将 I 分支 layer4 的输出（planes * 8）压缩到 planes * 2
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        # pag3 & pag4：使用 PagFM 模块进行融合
        # 它让 P 分支（细节特征）根据注意机制，有选择地学习 I 分支（语义特征）的信息
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        # layer3_：处理第一次融合后的特征，包含 m 个 BasicBlock，分辨率保持 1/8
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # layer4_：处理第二次融合后的特征，包含 m 个 BasicBlock，分辨率保持 1/8
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # layer5_：最后的特征细化层，使用 1 个 Bottleneck 块
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            # layer3_d & layer4_d：D 分支的初始特征处理层，保持 1/8 分辨率
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)

            # diff3 & diff4：特征差分模块（核心）
            # 它们从 I 分支的深层特征（planes * 4 / 8）中提取空间变化信息，模拟“求导”过程来定位边界
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )

            # spp：使用 PAPPM（轻量级金字塔池化），增强 D 分支对全局特征的理解
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            # dfm：使用 Light_Bag（轻量级边界注意引导融合），准备最后的三路合一
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            # 相比轻量版，此处使用了更多的 planes * 2 通道，且使用了更强大的 DAPPM 和标准 Bag 模块
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

        # layer5_d：D 分支最后的特征提取层，使用 Bottleneck 结构进一步精炼边界特征
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
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

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


if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
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
    print(FPS)





