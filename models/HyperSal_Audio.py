import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict

from models.ConditionNet import ConditionNet

import torch.nn.init as init

from models.HyperSal import DecoderConvUp

from models.soundnet import SoundNet


class AudioVideoSaliencyModel(nn.Module):
    def __init__(self, pretrain=None, use_support_Mod=True, residual_fusion=False, fix_soundnet=False,
                 hyper_type='multiply', test_mode=False):
        super(AudioVideoSaliencyModel, self).__init__()

        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.decoder = DecoderConvUp(hyper_type=hyper_type, use_support_Mod=use_support_Mod)

        self.use_support_Mod = use_support_Mod

        if self.use_support_Mod:
            self.conv_index = '22'
            self.support_size = 32 # nmber of input frames

            self.condition_net = ConditionNet(conv_index=self.conv_index, support_size=self.support_size)
            
        self.audionet = SoundNet()
        if not test_mode: # only for training mode
            self.audionet.load_state_dict(torch.load('./soundnet8_final.pth'))
            print("Loaded SoundNet Weights")
            
            n_param = 0
            for param in self.audionet.parameters():
                if fix_soundnet:
                    param.requires_grad = False
                    n_param += 1
                else:
                    param.requires_grad = True
                    # print(">>> train soundnet !!!")
            print("Fixing SoundNet: ", fix_soundnet, 'param.requires_grad=False- n_param: ', n_param)
            # tt

        self.audio_conv_1x1 = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, bias=True)
        
        self.maxpool = nn.MaxPool3d((16,1,1),stride=(2,1,2),padding=(0,0,0))
        self.bilinear = nn.Bilinear(6*7, 3, 16*7*12)
        self.residual_fusion = residual_fusion

        self.relu = nn.ReLU()
        # self.bn3d = nn.BatchNorm3d(768)
        # self.bn2d_audio = nn.BatchNorm2d(768)

        # self.fusion = nn.Conv3d(in_channels=768, out_channels=768, kernel_size=1, stride=1, bias=True)

        print(">>> self.residual_fusion in AudioVideoSaliencyModel: ", self.residual_fusion)


    def forward(self, x, audio, reurn_audio=False):
        if self.use_support_Mod:
            # concate x along time axix of x with shape of (B, C, T, H, W) = (1, 3, 32, 224, 384)
            B, C, T, H, W = x.shape
            x1 = x.reshape(B, C*T, H, W) # [1, 96, 224, 384]

            condition_weight = self.condition_net(x1) # condition_weight: (1, 128, 1, 1)

        if reurn_audio:
            # for debug
            x, [y1, y2, y3, y4], middle_features = self.backbone(x, return_fea=True)
        else:
            x, [y1, y2, y3, y4] = self.backbone(x) # x: (1, 768, 16, 7, 12)
           
        if self.residual_fusion:
            x2 = x.clone()

        # ml for debug
        # audio = torch.zeros_like(audio); print(">>> audio is set to zero !!!")

        audio_soundnet = self.audionet(audio)
        audio = self.relu(self.audio_conv_1x1(audio_soundnet)) # (1, 768, 3, 1)
        # print(">>> audio shape: ", audio.shape); tt
        x2 = self.maxpool(x2)

        # fuse visual and audio
        x2 = self.bilinear(x2.flatten(2), audio.flatten(2))
        x2 = x2.view(x2.size(0), x2.size(1), 16, 7, 12) # bcthw-(1, 768, 16, 7, 12)
        # print(">>> x1 shape: ", x1.shape); tt
    
        if self.residual_fusion:
            x = x + x2

        if reurn_audio:
            return self.decoder(x, y3, y2, y1, condition_weight), {'audio': audio, 
                                                                   'audio_soundnet': audio_soundnet,
                                                                   'encoder': [x, y3, y2, y1, condition_weight],
                                                                   'middle_features': middle_features}
        else:
            if self.use_support_Mod:
                return self.decoder(x, y3, y2, y1, condition_weight)
            else:
                return self.decoder(x, y3, y2, y1)
        