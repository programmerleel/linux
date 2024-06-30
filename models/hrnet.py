# -*- coding: utf-8 -*-
# @Time    : 2024/6/28/028 18:00
# @Author  : Shining
# @File    : hrnet.py
# @Description :

import torch
from torch import nn

import utils.config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(out)
        out = out + residual
        out = self.relu(out)
        return out


class StemNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class BranchBlock(nn.Module):
    def __init__(self, branch_num, num_in_channels, num_channels, block, block_num):
        super().__init__()
        self.branch_num = branch_num
        self.num_in_channels = num_in_channels
        self.num_channels = num_channels
        self.block = block
        self.block_num = block_num
        self.branches = self._make_branches()

    def _make_one_branch(self, branch_index, stride=1):
        downsample = None
        # 统一通道 当使用BottleBeck时 配置文件中的NUM_CHANNELS 适当的减小
        # 获取上一层的通道数
        if stride != 1 or self.num_in_channels[branch_index] != self.num_channels[branch_index]:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.num_in_channels[branch_index],
                          out_channels=self.num_channels[branch_index] * self.block.expansion, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.num_channels[branch_index] * self.block.expansion, momentum=0.1)
            )
        layers = []
        # 输入输出通道不相等 残差默认加上输入 不使用downsample统一通道 会出现错误
        layers.append(
            self.block(self.num_in_channels[branch_index], self.num_channels[branch_index] * self.block.expansion,
                       stride=1, downsample=downsample))
        self.num_in_channels[branch_index] = self.num_channels[branch_index] * self.block.expansion
        for i in range(1, self.block_num[branch_index]):
            layers.append(
                self.block(self.num_in_channels[branch_index], self.num_channels[branch_index] * self.block.expansion))

        return nn.Sequential(*layers)

    def _make_branches(self):
        branches = []
        for i in range(self.branch_num):
            branches.append(self._make_one_branch(i))

        return nn.ModuleList(branches)

    def forward(self, x):
        if self.branch_num == 1:
            return [self.branches[0](x)]

        branches_out = []
        for i in range(self.branch_num):
            branches_out.append(self.branches[i](x[i]))
        return branches_out


class TransitionBlock(nn.Module):
    def __init__(self, pre_branch_num, branch_num, num_in_channels, num_channels):
        super().__init__()
        self.pre_branch_num = pre_branch_num
        self.branch_num = branch_num
        self.num_in_channels = num_in_channels
        self.num_channels = num_channels
        self.transitions = self._make_transition()

    def _make_transition(self):
        transition_layers = []
        for i in range(self.pre_branch_num):
            transition_layer = []
            for j in range(self.branch_num):
                if i == j:
                    transition_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=self.num_channels[j], out_channels=self.num_in_channels[j],
                                      kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(self.num_in_channels[j]),
                            nn.ReLU(inplace=True)
                        )
                    )
                elif j > i:
                    # for k in range((j-i)):
                    # if k == 0:
                    transition_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=self.num_channels[i], out_channels=self.num_in_channels[j],
                                      kernel_size=3, stride=2 ** (j - i), padding=1, bias=False),
                            nn.BatchNorm2d(self.num_in_channels[j]),
                            nn.ReLU(inplace=True)
                        )
                    )

                elif j < i:
                    # for k in range(i - j):
                    # if k == 0:
                    transition_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=self.num_channels[i], out_channels=self.num_in_channels[j],
                                      kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(self.num_in_channels[j]),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2 ** (i - j), mode='nearest')
                        )
                    )

            transition_layers.append(nn.ModuleList(transition_layer))
        return nn.ModuleList(transition_layers)

    def _fuse_layer(self, transition_outs):

        if self.pre_branch_num == 1:
            return transition_outs[0]
        else:
            fuse_outs = []
            transition_outs_ = []
            for i in range(self.pre_branch_num):
                for j in range(self.branch_num):
                    transition_outs_.append(transition_outs[i][j])
            for k in range(self.branch_num):
                fuse_out = transition_outs_[k] + transition_outs_[k + self.branch_num]
                fuse_outs.append(fuse_out)
            return fuse_outs

    def forward(self, x):
        transition_outs = []
        for i in range(self.pre_branch_num):
            transition_out = []
            for j in range(self.branch_num):
                transition_out.append(self.transitions[i][j](x[i]))
            transition_outs.append(transition_out)
        transition_outs = self._fuse_layer(transition_outs)
        return transition_outs


BLOCK = {
    "BASIC": BasicBlock,
    "BOTTLE": BottleNeck
}


class HRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem_net = StemNet(3, config["MODEL"]["STRUCTURE"]["STEM_CHANNELS"][0])
        self.stage1 = []
        for i in range(config["MODEL"]["STRUCTURE"]["STAGE1"]["MODULE_NUM"]):

            branch = BranchBlock(branch_num=config["MODEL"]["STRUCTURE"]["STAGE1"]["BRANCH_NUM"],
                                 num_in_channels=config["MODEL"]["STRUCTURE"]["STEM_CHANNELS"],
                                 num_channels=config["MODEL"]["STRUCTURE"]["STAGE1"]["NUM_CHANNELS"],
                                 block_num=config["MODEL"]["STRUCTURE"]["STAGE1"]["BLOCK_NUM"],
                                 block=BLOCK[config["MODEL"]["STRUCTURE"]["STAGE1"]["BLOCK"]])
            self.stage1.append(branch)
            if i != config["MODEL"]["STRUCTURE"]["STAGE1"]["MODULE_NUM"] - 1:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE1"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE1"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STEM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE1"]["NUM_CHANNELS"])
                self.stage1.append(transition)
            else:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE1"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE1"]["NUM_CHANNELS"])
                self.stage1.append(transition)

        self.stage2 = []
        for i in range(config["MODEL"]["STRUCTURE"]["STAGE2"]["MODULE_NUM"]):

            branch = BranchBlock(branch_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BRANCH_NUM"],
                                 num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"],
                                 num_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"],
                                 block_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BLOCK_NUM"],
                                 block=BLOCK[config["MODEL"]["STRUCTURE"]["STAGE2"]["BLOCK"]])
            self.stage2.append(branch)
            if i != config["MODEL"]["STRUCTURE"]["STAGE2"]["MODULE_NUM"] - 1:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"])
                self.stage2.append(transition)
            else:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE2"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE2"]["NUM_CHANNELS"])
                self.stage2.append(transition)

        self.stage3 = []
        for i in range(config["MODEL"]["STRUCTURE"]["STAGE3"]["MODULE_NUM"]):

            branch = BranchBlock(branch_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BRANCH_NUM"],
                                 num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"],
                                 num_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"],
                                 block_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BLOCK_NUM"],
                                 block=BLOCK[config["MODEL"]["STRUCTURE"]["STAGE3"]["BLOCK"]])
            self.stage3.append(branch)
            if i != config["MODEL"]["STRUCTURE"]["STAGE3"]["MODULE_NUM"] - 1:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"])
                self.stage3.append(transition)
            else:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE3"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE3"]["NUM_CHANNELS"])
                self.stage3.append(transition)

        self.stage4 = []
        for i in range(config["MODEL"]["STRUCTURE"]["STAGE4"]["MODULE_NUM"]):

            branch = BranchBlock(branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                 num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"],
                                 num_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"],
                                 block_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BLOCK_NUM"],
                                 block=BLOCK[config["MODEL"]["STRUCTURE"]["STAGE3"]["BLOCK"]])
            self.stage4.append(branch)
            if i != config["MODEL"]["STRUCTURE"]["STAGE4"]["MODULE_NUM"] - 1:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"])
                self.stage4.append(transition)
            else:
                transition = TransitionBlock(pre_branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                             branch_num=config["MODEL"]["STRUCTURE"]["STAGE4"]["BRANCH_NUM"],
                                             num_in_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"],
                                             num_channels=config["MODEL"]["STRUCTURE"]["STAGE4"]["NUM_CHANNELS"])
                self.stage4.append(transition)

    def _make_layer(self, block, in_channels, out_channels, stride=1):
        downsample = None
        # 统一通道 当使用BottleBeck时 配置文件中的NUM_CHANNELS 适当的减小
        # 获取上一层的通道数
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * block.expansion, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion, momentum=0.1)
            )

        return block(in_channels, out_channels * block.expansion,
                     stride=1, downsample=downsample)

    def _fuse_layer(self, outs):
        cat_out = 0
        for i in range(len(outs)):
            if i == 0:
                cat_out = outs[i]
            else:
                cat_out = torch.cat([cat_out, nn.Upsample(scale_factor=2 ** i, mode='nearest')(outs[i])], dim=1)
        up_out1 = self._make_layer(BLOCK[config["MODEL"]["STRUCTURE"]["CAT"]["BLOCK"]], cat_out.shape[1],
                                config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"])(cat_out)
        up_out2 = self._make_layer(BLOCK[config["MODEL"]["STRUCTURE"]["CAT"]["BLOCK"]], config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"],
                              config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"])(
                                  nn.Upsample(scale_factor=2, mode='nearest')(up_out1))
        up_out3 = self._make_layer(BLOCK[config["MODEL"]["STRUCTURE"]["CAT"]["BLOCK"]], config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"],
                              config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"])(
                                  nn.Upsample(scale_factor=2, mode='nearest')(up_out2))
        return up_out3

    def _make_head(self,out):
        class_num = config["MODEL"]["STRUCTURE"]["SEG_HEAD"]["CLASS_NUM"]
        kpt_num = config["MODEL"]["STRUCTURE"]["KPT_HEAD"]["KPT_NUM"]
        seg_out = self._make_layer(BLOCK[config["MODEL"]["STRUCTURE"]["SEG_HEAD"]["BLOCK"]], config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"],class_num)(out)
        kpt_out = self._make_layer(BLOCK[config["MODEL"]["STRUCTURE"]["KPT_HEAD"]["BLOCK"]], config["MODEL"]["STRUCTURE"]["CAT"]["NUM_CHANNELS"],kpt_num)(out)
        return seg_out,kpt_out

    def forward(self, x):
        stem_out = self.stem_net(x)
        stage1_out, stage2_out, stage3_out, stage4_out = 0, 0, 0, 0
        for i in range(len(self.stage1)):
            if i == 0:
                stage1_out = self.stage1[i](stem_out)
            else:
                stage1_out = self.stage1[i](stage1_out)

        for i in range(len(self.stage2)):
            if i == 0:
                stage2_out = self.stage2[i](stage1_out)
            else:
                stage2_out = self.stage2[i](stage2_out)

        for i in range(len(self.stage3)):
            if i == 0:
                stage3_out = self.stage3[i](stage2_out)
            else:
                stage3_out = self.stage3[i](stage3_out)

        for i in range(len(self.stage4)):
            if i == 0:
                stage4_out = self.stage4[i](stage3_out)
            else:
                stage4_out = self.stage4[i](stage4_out)

        fuse_out = self._fuse_layer(stage4_out)
        seg_out,kpt_out = self._make_head(fuse_out)
        return seg_out,kpt_out


if __name__ == '__main__':
    x = torch.randn(2, 3, 640, 640)

    config = utils.config.load_config(r"C:\Users\Windows\Desktop\HRNet-Pose-Segmentation\configs\base_config.yaml")
    model = HRNet(config)
    seg_out,kpt_out = model.forward(x)
    print(seg_out.shape)
    print(kpt_out.shape)

