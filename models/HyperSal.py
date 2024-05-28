import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict

from models.ConditionNet import ConditionNet

import torch.nn.init as init

class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, rgb_fea):
        gate = torch.sigmoid(self.gate(rgb_fea))
        gate_fea = rgb_fea * gate + rgb_fea

        return gate_fea

class Base_Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, use_support_Mod, hyper_type):
        super(Base_Conv3D, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size[0], 
                                               kernel_size[1], kernel_size[2])) # (1, 24, 1, 3, 3)
        self.scale = 1 / math.sqrt(in_channel * kernel_size[0] * kernel_size[1] * kernel_size[2])

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding

        self.use_support_Mod = use_support_Mod

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

        self.hyper_type = hyper_type
        print("self.hyper_type: ", self.hyper_type)
        
    def forward(self, input, condition_feature=None):
        
        # print("input.shape: ", input.shape); tt # [1, 24, 1, 224, 384]
        b, c, T, h, w = input.shape

        if c != self.in_channel:
            raise ValueError('Input channel is not equal with conv in_channel')
        
        if self.use_support_Mod == True: # and condition_feature != None
            #[batch, out_channel, in_channel, self.kernel_size, self.kernel_size] = [batch, 64, 64, 3, 3]
            if self.hyper_type == 'multiply':
                weight = self.weight.unsqueeze(0) * self.scale * condition_feature
            elif self.hyper_type == 'add':
                weight = self.weight.unsqueeze(0) + condition_feature
            else:
                tt
            
            # weight = weight.view(b*self.in_channel, self.out_channel, T, 
            #                      self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]) # (1, 24, 1, 3, 3)
            weight = weight.view(b*self.out_channel, self.in_channel,
                                 self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]) # (1, 24, 1, 3, 3)
            input = input.view(1, b*self.in_channel, T, h, w)
            bias = torch.repeat_interleave(self.bias, repeats=b, dim=0) if self.bias else None
            out = F.conv3d(input,weight,bias=bias,stride=self.stride,padding=self.padding,groups=b)

            # print("out.shape: ", out.shape); tt # [1, 1, 1, 224, 384]
            _, _, T1, height, width = out.shape
            out = out.view(b, self.out_channel, T1, height, width)
        else:
            tt

        return out

class Conv3d_Module(nn.Module):
    def __init__(self, in_plane=192, use_support_Mod=True, layer_index='layer1', bias=False, hyper_type=None):
        super(Conv3d_Module, self).__init__()

        self.use_support_Mod = use_support_Mod

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')

        self.conv3d_module1 = nn.Sequential(
            nn.Conv3d(in_plane, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2 if layer_index in ['layer2', 'layer3', 'layer4'] else nn.Identity()
        )

        if self.use_support_Mod:
            self.conv3d_module2 = Base_Conv3D(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), 
                                            padding=(0, 1, 1), bias=False, use_support_Mod=use_support_Mod,
                                            hyper_type=hyper_type)
        else:
            self.conv3d_module2 = nn.Sequential(
                nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                # Base_Conv3D(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, use_support_Mod=use_support_Mod),
                nn.Sigmoid()
            )
        
        
        if layer_index in ['layer1', 'layer2']:
            self.upsample_conv4 = nn.Identity()
        elif layer_index in ['layer3']:
            self.upsample_conv4 = self.upsampling2
        elif layer_index in ['layer4']:
            self.upsample_conv4 = self.upsampling4
        else:
            tt

        self.sigmoid = nn.Sigmoid()



    def forward(self, x, condition_weight=None):
        x = self.conv3d_module1(x)
        if self.use_support_Mod:
            x = self.conv3d_module2(x, condition_weight[0]) # applied for conv0 --  only applied to one conv for each block
        else:
            x = self.conv3d_module2(x)
        x = self.sigmoid(self.upsample_conv4(x))

        return x


class VideoSaliencyModel_ml(nn.Module):
    def __init__(self, pretrain=None, use_support_Mod=True, hyper_type='multiply'):
        super(VideoSaliencyModel_ml, self).__init__()
        
        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.decoder = DecoderConvUp(hyper_type=hyper_type, use_support_Mod=use_support_Mod)

        print(">>> hyper_type: ", hyper_type)

        self.use_support_Mod = use_support_Mod

        if self.use_support_Mod:
            self.conv_index = '22'
            self.support_size = 32 # nmber of input frames

            self.condition_net = ConditionNet(conv_index=self.conv_index, support_size=self.support_size)
            
    def forward(self, x, reurn_audio=False):
        if self.use_support_Mod:
            # concate x along time axix of x with shape of (B, C, T, H, W) = (1, 3, 32, 224, 384)
            B, C, T, H, W = x.shape
            x1 = x.reshape(B, C*T, H, W) # [1, 96, 224, 384]

            condition_weight = self.condition_net(x1) # condition_weight: (1, 128, 1, 1)

        x, [y1, y2, y3, y4] = self.backbone(x)


        if reurn_audio:
            return self.decoder(x, y3, y2, y1, condition_weight), {'audio': [], 
                                                                #    'audio_soundnet': audio_soundnet,
                                                                   'encoder': [x, y3, y2, y1, condition_weight],
                                                                #    'middle_features': middle_features
                                                                   }
        else:
            tt
            return self.decoder(x, y3, y2, y1, condition_weight)


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

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        input = input.view(input.shape[0],-1)
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

class Modulations(nn.Module):
    def __init__(self, n_block =10, n_conv_each_block=2, conv_index='22', sr_in_channel=64):
        super(Modulations, self).__init__()
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block*n_conv_each_block
        self.conv_index = conv_index
        if conv_index == '22':
            self.condition_channel = 128
        elif self.conv_index == '54':
            self.condition_channel = 256
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        
        self.sr_in_channel=sr_in_channel

        self.modulations = self.get_linear_modulations()
        initialize_weights(self.modulations,0.1)
    
    def get_linear_modulations(self):
        modules = []
        for _ in range(self.n_modulation):
            modules.append(EqualLinear(self.condition_channel, self.sr_in_channel, bias_init=1))
        
        return nn.Sequential(*modules)
    
    def forward(self, condition_feature):
        '''
        Input:
        For training
        condition_feature:[task_size, 128, 1, 1]
        For testing
        condition_feature:[1, 128, 1, 1]

        repeat n_block*2 condition_features [n_block*n_conv_each_block, task_size, 128, 1, 3, 3]
        for i in range n_block*2:
            EqualLinear modulation condition_features[i] [task_size, 1, 64, 1, 1]
        condition_features [n_block, n_conv_each_block, task_size, 1, 64, 1, 1]
        '''
        task_size, condition_channel, h, w = condition_feature.shape
        if condition_channel != self.condition_channel:
            raise ValueError('the shape of input condition_feature should be [task_size, condition_channel, h, w]')

        condition_weight = []
        repeat_support_feature = torch.repeat_interleave(condition_feature.unsqueeze(0), repeats=self.n_modulation, dim=0)#[n_block*2, task_size, 128, 1, 1]
        for idx, modulation in enumerate(self.modulations):
            cur_support_feature = repeat_support_feature[idx]
            reshape_condition_feature = modulation(cur_support_feature.permute(0, 2, 3, 1)).view(task_size, 1, self.sr_in_channel, 1, 1)
            condition_weight.append(reshape_condition_feature.unsqueeze(0)) 
        
        out_features = torch.cat(condition_weight, 0).to(condition_feature.device)
        # out_features = out_features.view(self.n_block, self.n_conv_each_block, task_size, 1, self.sr_in_channel, 1, 1)
        out_features = out_features.view(self.n_block, self.n_conv_each_block, task_size, 1, self.sr_in_channel, 1, 1, 1)
        
        return out_features

class DecoderConvUp(nn.Module):
    def __init__(self, use_support_Mod=True, hyper_type=None):
        super(DecoderConvUp, self).__init__()

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

        self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.convtsp1 = Conv3d_Module(192, use_support_Mod=use_support_Mod, layer_index='layer1', hyper_type=hyper_type)
        self.convtsp2 = Conv3d_Module(192, use_support_Mod=use_support_Mod, layer_index='layer2', hyper_type=hyper_type)
        self.convtsp3 = Conv3d_Module(192, use_support_Mod=use_support_Mod, layer_index='layer3', hyper_type=hyper_type)
        self.convtsp4 = Conv3d_Module(192, use_support_Mod=use_support_Mod, layer_index='layer4', hyper_type=hyper_type)
        self.use_support_Mod = use_support_Mod

        # ml 
        self.n_block = 4
        self.n_conv_each_block = 1
        self.channels = 24 # number of channels applied the condition feature

        self.conv_index = '22' # keep default
        
        self.modulations = Modulations(n_block =self.n_block, n_conv_each_block=self.n_conv_each_block, 
                                       conv_index=self.conv_index, sr_in_channel=self.channels)

        self.convout = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

        self.gate1 = Gate(192)
        self.gate2 = Gate(192)
        self.gate3 = Gate(192)
        self.gate4 = Gate(192)

    def forward(self, y4, y3, y2, y1, condition_feature=None):
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        t3 = self.upsampling2(y4) + y3
        y3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + y2 + self.upsampling4(y4)
        y2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + y1 + self.upsampling8(y4)
        y1 = self.convs1(t1)

        y1 = self.gate1(y1)
        y2 = self.gate2(y2)
        y3 = self.gate3(y3)
        y4 = self.gate4(y4)

        if self.use_support_Mod:
            b, c, T, h, w = y4.shape
            condition_feature = self.modulations(condition_feature) #[n_block, n_conv_each_block, task_size, 1, 64, 1, 1]
            condition_feature = torch.repeat_interleave(condition_feature, repeats=b//condition_feature.shape[2], dim=2) #[n_block, n_conv_each_block, batch, 1, 64, 1, 1]
        
            z1 = self.convtsp1(y1, condition_feature[0]) # for block 0
            z2 = self.convtsp2(y2, condition_feature[1]) # for block 1
            z3 = self.convtsp3(y3, condition_feature[2]) # for block 2
            z4 = self.convtsp4(y4, condition_feature[3]) # for block 3
        else:
            z1 = self.convtsp1(y1) # for block 0
            z2 = self.convtsp2(y2) # for block 1
            z3 = self.convtsp3(y3) # for block 2
            z4 = self.convtsp4(y4) # for block 3


        z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))

        z0 = z0.view(z0.size(0), z0.size(3), z0.size(4))
        
        return z0
