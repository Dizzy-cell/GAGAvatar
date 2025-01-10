
import math
import torch
from torch import nn
from torch.nn import functional as F

class Unet_rgb(nn.Module):
    def __init__(
        self, in_size = 512, out_size = 512, in_dim = 32, out_dim=3, 
        num_style_feat=512, num_mlp=8, activation=True,
    ):
        super().__init__()
        self.activation = activation
        self.num_style_feat = num_style_feat
        self.in_size, self.out_size = in_size, out_size
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        assert in_size <= out_size*2, f'In/out: {in_size}/{out_size}.'
        assert f'{in_size}' in channels.keys(), f'In size: {in_size}.'
        assert f'{out_size}' in channels.keys(), f'Out size: {out_size}.'
        self.log_size = int(math.log(out_size, 2))
        ### UNET Module
        if self.in_size <= self.out_size:
            self.conv_body_first = nn.Conv2d(in_dim, channels[f'{out_size}'], 1)
        else:
            self.conv_body_first = nn.ModuleList([
                nn.Conv2d(in_dim, channels[f'{in_size}'], 1),
                ResBlock(channels[f'{in_size}'], channels[f'{out_size}'], mode='down'),
            ])
        # downsample
        in_channels = channels[f'{out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)
        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels
        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f'{2**i}'], 3, 1))

    def forward(self, x, randomize_noise=True):
        conditions, unet_skips, out_rgbs = [], [], []
        # size
        # UNET downsample
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)        #(B,16,512,512) 

        for i in range(self.log_size - 2):                                    #9
            feat = self.conv_body_down[i](feat)                                       # 32 , 256, 256 
            unet_skips.insert(0, feat)                                                # 64, 128, 128
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)               # 256x4x4

        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)

        image = self.toRGB[self.log_size - 3](feat)
        # activation
        if self.activation:
            image = torch.sigmoid(image)
            # image = image*(1 + 2*0.001) - 0.001 
        # print("Unet_rgb",219)
        # from IPython import embed 
        # embed()  
        return image#, out_rgbs
class ResBlock(nn.Module):
    
    Residual block with bilinear upsampling/downsampling.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode. Options: down | up. Default: down.
    
    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )
        skip = self.skip(x)
        out = out + skip
        return out
if __name__ == '__main__':
    import time
    with torch.no_grad():
        model = Unet_rgb(in_size=512, in_dim=32, out_dim=3, out_size=512).cuda()
        model.eval()
        from tqdm import tqdm
        data = torch.rand(2, 3, 512, 512).cuda()
        print(model(data.clone()).shape)
        import ipdb; ipdb.set_trace()
        for i in tqdm(range(100)):
            result = model(data.clone())
