import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ConditionNet(nn.Module):
    def __init__(self, conv_index='22', support_size=10):
        super(ConditionNet, self).__init__()
        self.support_size = support_size

        self.conv_index = conv_index
        if conv_index == '22':
            self.condition_channel = 128
        elif self.conv_index == '54':
            self.condition_channel = 256
        else:
            raise ValueError('Illegal VGG conv_index!!!')

        self.condition = self.get_VGG_condition()
        initialize_weights(self.condition,0.1)

    def get_VGG_condition(self):
        
        cfg = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
        if self.conv_index == '22':
            cfg_idx = cfg[:5]
        elif self.conv_index == '54':
            cfg_idx = cfg[:35]
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        return self._make_layers(cfg_idx)
    
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3*self.support_size
        for v in cfg:
            if v == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def reset_support_size(self, support_size):
        self.support_size = support_size

    def forward(self, support_x):
        '''return task_size condition_features
        Input:
        For training
        support_x [task_size, support_size*3, h, w]
        For testing task_size = 1
        support_x [1, support_size*3, h, w]
        
        '''
        support_conditional_feature = self.condition(support_x) #[task_size, 128, h/2, w/2]

        _, _, h, w = support_conditional_feature.shape
        conditional_feature = F.avg_pool2d(support_conditional_feature, kernel_size=h, stride=w)#[task_size, 128, 1, 1]
        return conditional_feature