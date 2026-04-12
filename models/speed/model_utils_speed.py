# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 生成综合的注意力权重图 (包含空间分布坐标信息与通道权重)
        att = a_h * a_w
        # 返回加权后的特征 以及 生成的注意力权重，用于跨层级动态注入
        return identity * att, att
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(x))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=algc)

        return out


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear', align_corners=algc) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear', align_corners=algc) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear', align_corners=algc) + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear', align_corners=algc) + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class AtrousBranch(nn.Module):
    """
    使用空洞卷积的分支，专门捕捉不规则、多尺度的局部结构
    """
    def __init__(self, inplanes, branch_planes, dilation, BatchNorm=nn.BatchNorm2d, bn_mom=0.1):
        super(AtrousBranch, self).__init__()
        self.conv = nn.Sequential(
            # kernel=3, stride=1, 利用 dilation 扩大感受野，padding=dilation 保持尺寸不变
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        # self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        #                             BatchNorm(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        # self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        #                             BatchNorm(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        # self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
        #                             BatchNorm(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )


        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )

        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out
class PAPPM_optimized(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM_optimized, self).__init__()
        bn_mom = 0.1

        # 0. 局部最细粒度分支
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        # 1-3. 空洞卷积分支 (替换原有的 Average Pooling)
        # dilation 设为 2, 4, 8，相当于“网”越撒越大，能覆盖各种弯曲角度的裂缝
        self.scale1 = AtrousBranch(inplanes, branch_planes, dilation=2, BatchNorm=BatchNorm, bn_mom=bn_mom)
        self.scale2 = AtrousBranch(inplanes, branch_planes, dilation=4, BatchNorm=BatchNorm, bn_mom=bn_mom)
        self.scale3 = AtrousBranch(inplanes, branch_planes, dilation=8, BatchNorm=BatchNorm, bn_mom=bn_mom)

        # 4. 全局特征分支 (保留，获取图像级背景先验)
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale_process = nn.Sequential(
            BatchNorm(branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)

        # 空洞卷积不会改变尺寸，直接与 x_ 叠加
        scale_list.append(self.scale1(x) + x_)
        scale_list.append(self.scale2(x) + x_)
        scale_list.append(self.scale3(x) + x_)

        # 只有全局池化需要上采样恢复尺寸
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


# class PagFM(nn.Module):
#     """
#     针对细长裂缝优化的 PagFM
#     引入了 3x3 深度可分离卷积获取局部空间感受野，以及温度锐化系数应对 94:6 的极度不平衡
#     """
#
#     def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
#         super(PagFM, self).__init__()
#         self.with_channel = with_channel
#         self.after_relu = after_relu
#
#         # 【优化1】：温度锐化系数。可学习参数，初始倍数设为 2.0
#         # 用于将微弱的相似度特征放大，防止因为 94:6 的背景压制导致注意力均值趋近于0
#         self.temp = nn.Parameter(torch.ones(1) * 2.0)
#
#         # 【优化2】：使用 3x3 深度可分离卷积替换 1x1 卷积
#         # 深度可分离 = 3x3 Depthwise (捕捉空间连贯性) + 1x1 Pointwise (跨通道映射)
#         self.f_x = nn.Sequential(
#             # Depthwise Conv
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
#                       groups=in_channels, bias=False),
#             # Pointwise Conv
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
#             BatchNorm(mid_channels)
#         )
#         self.f_y = nn.Sequential(
#             # Depthwise Conv
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
#                       groups=in_channels, bias=False),
#             # Pointwise Conv
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
#             BatchNorm(mid_channels)
#         )
#         if with_channel:
#             self.up = nn.Sequential(
#                 nn.Conv2d(mid_channels, in_channels,
#                           kernel_size=1, bias=False),
#                 BatchNorm(in_channels)
#             )
#         if after_relu:
#             self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, y):
#         input_size = x.size()
#         if self.after_relu:
#             y = self.relu(y)
#             x = self.relu(x)
#
#         # y_q: 经过局部感受野后的 Query
#         y_q = self.f_y(y)
#         y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
#                             mode='bilinear', align_corners=False)
#         # x_k: 经过局部感受野后的 Key
#         x_k = self.f_x(x)
#
#         if self.with_channel:
#             # 引入 self.temp 进行锐化
#             sim_map = torch.sigmoid(self.up(x_k * y_q) * self.temp)
#         else:
#             # 引入 self.temp 进行锐化
#             sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1) * self.temp)
#
#         y = F.interpolate(y, size=[input_size[2], input_size[3]],
#                           mode='bilinear', align_corners=False)
#         x = (1 - sim_map) * x + sim_map * y
#
#         return x

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels,
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y

        return x
# class Light_Bag(nn.Module):
#     """
#     针对极度不平衡裂缝数据优化的 Light_Bag
#     结合了：注意力锐化、通道筛选 (SE) 以及高分辨率特征补偿
#     """
#
#     def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
#         super(Light_Bag, self).__init__()
#
#         # 1. 注意力锐化系数：可学习参数，初始放大倍数设为 2.0
#         # 用于将微弱的裂缝边界预测 d 放大，防止经过 sigmoid 后全部趋近于 0
#         self.temp = nn.Parameter(torch.ones(1) * 2.0)
#
#         self.conv_p = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels,
#                       kernel_size=1, bias=False),
#             BatchNorm(out_channels)
#         )
#         self.conv_i = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels,
#                       kernel_size=1, bias=False),
#             BatchNorm(out_channels)
#         )
#
#         # 2. 通道注意力机制 (SE Block)
#         # 针对 94:6 的比例，在通道维度动态抑制背景，增强裂缝特征
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(out_channels, max(1, out_channels // 4), 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(max(1, out_channels // 4), out_channels, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, p, i, d):
#         # 【优化 1】：注意力锐化
#         edge_att = torch.sigmoid(d * self.temp)
#
#         # 保留原有的交叉注意力引导逻辑
#         p_add = self.conv_p((1 - edge_att) * i + p)
#         i_add = self.conv_i(i + edge_att * p)
#
#         # 融合 P 分支(上下文)与 I 分支(细节)
#         fused = p_add + i_add
#
#         # 【优化 2】：通道注意力筛选
#         fused = fused * self.se(fused)
#
#         # 【优化 3】：高频特征补偿
#         # 将高分辨率分支(i)的原始特征作为残差加回，防止细微裂缝断裂
#         # (因为 PIDNet 传入 Light_Bag 时 in_channels == out_channels，可直接相加)
#         return fused + i
class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()

        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels, momentum=bn_mom)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels, momentum=bn_mom)
        )
        # self.cross_attention = CrossAttention(out_channels)


    def forward(self, p, i, d):
        # 1. 原始的基于 D 分支的特征加权
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        # 2. 初步融合得到 F_Bag
        f_bag = p_add + i_add

        # 3. 通过交叉注意力机制动态调整特征间的关联 (F_Cross = F_Bag + O)
        # f_cross = self.cross_attention(f_bag)
        return f_bag

class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
            # BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            # BatchNorm(out_channels)
        )
        self.conv_i = nn.Sequential(
            # BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            # BatchNorm(out_channels)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            # BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


if __name__ == '__main__':
    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()

    out = net(x, y)
