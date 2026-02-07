# ==============================================================================
# CHAOSNET: CUSTOM NEURAL ARCHITECTURE (Built From Scratch)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChaosBlock(nn.Module):
    """
    Custom 'Chaos' Block - A novel efficient feature extractor.
    combines Depthwise Separable Convolutions with Channel Shuffling.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        branch_c = out_c // 2
        
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, branch_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()
            
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c if stride > 1 else branch_c, branch_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_c, branch_c, 3, stride, 1, groups=branch_c, bias=False),
            nn.BatchNorm2d(branch_c),
            nn.Conv2d(branch_c, branch_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return self.channel_shuffle(out, 2)

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.data.size()
        channels_per_group = C // groups
        x = x.view(N, groups, channels_per_group, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(N, -1, H, W)
        return x

class Backbone(nn.Module):
    """
    The ChaosNet Backbone.
    A lightweight, high-performance feature extractor designed for Edge AI.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.stage2 = self._make_stage(24, 116, 4)
        self.stage3 = self._make_stage(116, 232, 8)
        self.stage4 = self._make_stage(232, 464, 4)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(464, num_classes)

    def _make_stage(self, in_c, out_c, repeat):
        layers = [ChaosBlock(in_c, out_c, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ChaosBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x)
        return features

