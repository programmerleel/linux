# -*- coding: utf-8 -*-
# @Time    : 2024/6/28/028 18:00
# @Author  : Shining
# @File    : hrnet.py
# @Description :

import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
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
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
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
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class BranchBlock(nn.Module):
    def __init__(self,branch_num,num_in_channels,num_channels,block,block_num):
        super().__init__()
        self.branch_num = branch_num
        self.num_in_channels = num_in_channels
        self.num_channels = num_channels
        self.block = block
        self.block_num = block_num
        self.branches = self._make_branches()

    def _make_one_branch(self,branch_index,stride=1):
        downsample = None
        # 统一通道 当使用BottleBeck时 配置文件中的NUM_CHANNELS 适当的减小
        # 获取上一层的通道数
        if stride != 1 or self.num_in_channels[branch_index] != self.num_channels[branch_index]:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.num_in_channels[branch_index],out_channels=self.num_channels[branch_index]*self.block.expansion,kernel_size=3,stride=stride,padding=1,bias=False),
                nn.BatchNorm2d(self.num_channels[branch_index]*self.block.expansion,momentum=0.1)
            )
        layers = []
        # 输入输出通道不相等 残差默认加上输入 不使用downsample统一通道 会出现错误
        layers.append(self.block(self.num_in_channels[branch_index],self.num_channels[branch_index]*self.block.expansion,stride=1,downsample=downsample))
        self.num_in_channels[branch_index] = self.num_channels[branch_index]*self.block.expansion
        for i in range(1,self.block_num[branch_index]):
            layers.append(self.block(self.num_in_channels[branch_index], self.num_channels[branch_index] * self.block.expansion))

        return nn.Sequential(*layers)

    def _make_branches(self):
        branches = []
        for i in range(self.branch_num):
            branches.append(self._make_one_branch(i))

        return nn.ModuleList(branches)

    def forward(self,x):
        if self.branch_num == 1:
            return [self.branches[0](x)]

        branches_out = []
        for i in range(self.branch_num):
            branches_out.append(self.branches[i](x[i]))
        return branches_out

class TransitionBlock(nn.Module):
    def __init__(self,pre_branch_num,branch_num,num_in_channels,num_channels):
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
                elif j>i:
                    # for k in range((j-i)):
                        # if k == 0:
                        transition_layer.append(
                            nn.Sequential(
                                nn.Conv2d(in_channels=self.num_channels[i],out_channels=self.num_in_channels[j],kernel_size=3,stride=2**(j-i),padding=1,bias=False),
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
                                nn.Upsample(scale_factor=2**(i-j), mode='nearest')
                            )
                        )

            transition_layers.append(nn.ModuleList(transition_layer))
        return nn.ModuleList(transition_layers)

    def _fuse_layer(self,transition_outs):

        if self.pre_branch_num == 1:
            return transition_outs[0]
        else:
            fuse_outs = []
            transition_outs_ = []
            for i in range(self.pre_branch_num):
                for j in range(self.branch_num):
                    transition_outs_.append(transition_outs[i][j])
            print(self.branch_num)
            for k in range(self.branch_num):
                fuse_out = transition_outs_[k] + transition_outs_[k+self.branch_num]
                print("xxx")
                print(fuse_out.shape)
                print("xxx")
                fuse_outs.append(fuse_out)
            return fuse_outs

    def forward(self, x):
        transition_outs = []
        for i in range(self.pre_branch_num):
            transition_out = []
            for j in range(self.branch_num):
                transition_out.append(self.transitions[i][j](x[i]))
                print("ooooooooooooooooooooooooooooooo")
                print(self.transitions[i][j])
                print(self.transitions[i][j](x[i]).shape)
            transition_outs.append(transition_out)
        return transition_outs



if __name__ == '__main__':
    x = torch.randn(2,3,640,640)

    stem_net = StemNet(3,64)
    stem_out = stem_net.forward(x)

    branch1 = BranchBlock(branch_num=1,num_in_channels=[64],num_channels=[64],block_num=[4],block=BasicBlock)
    branch1_out = branch1.forward(stem_out)
    transition1 = TransitionBlock(1,2,[32,64],[64])
    transition1_out = transition1.forward(branch1_out)
    fuse1_out = transition1._fuse_layer(transition1_out)
    for i in range(2):
        print(fuse1_out[i].shape)
    print("---------------------")

    branch2 = BranchBlock(branch_num=2, num_in_channels=[32,64,128], num_channels=[32,64], block_num=[4,4], block=BasicBlock)
    branch2_out = branch2.forward(fuse1_out)
    transition2 = TransitionBlock(2, 3, [32, 64,128], [32,64])
    transition2_out = transition2.forward(branch2_out)
    for i in range(2):
        for j in range(3):
            print(transition2_out[i][j].shape)
    fuse2_out = transition2._fuse_layer(transition2_out)
    for i in range(3):
        print(fuse2_out[i].shape)
    print("---------------------")

    branch3 = BranchBlock(branch_num=3, num_in_channels=[32, 64, 128,256], num_channels=[32, 64,128], block_num=[4, 4,4],block=BasicBlock)
    branch3_out = branch3.forward(fuse2_out)
    transition3 = TransitionBlock(3, 4, [32, 64, 128,256], [32, 64,128])
    transition3_out = transition3.forward(branch3_out)
    print(transition3)
    for i in range(3):
        for j in range(4):
            print(transition3_out[i][j].shape)
    fuse3_out = transition3._fuse_layer(transition3_out)

    branch4 = BranchBlock(branch_num=4, num_in_channels=[32, 64, 128,256], num_channels=[32, 64,128,256], block_num=[4, 4,4,4],block=BasicBlock)
    branch4_out = branch4.forward(fuse3_out)
    transition4 = TransitionBlock(4, 4, [32, 64, 128,256], [32, 64,128,256])
    transition4_out = transition4.forward(branch4_out)
    fuse4_out = transition4._fuse_layer(transition4_out)

    for j in range(4):
            print(fuse4_out[j].shape)


