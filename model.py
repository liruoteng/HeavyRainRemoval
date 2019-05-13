import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
from unet_parts import *
from torch.nn.modules.activation import Hardtanh
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from torch.nn import init
from networks import define_G, define_D
from torch.autograd import Variable


# -------------------------------------------------------------------------- #
#  Network Architecture
# -------------------------------------------------------------------------- #
class DecompModel(nn.Module):
    def __init__(self):
        super(DecompModel, self).__init__()
        self.rainnet = RainNet(33, 1)
        self.fognet = FogNet(33)
        self.radiux = [2, 4, 8, 16, 32]
        self.eps_list = [0.001, 0.0001]
        self.ref = RefUNet(3, 3)
        self.eps = 0.001
        self.gf = None
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        lf, hf = self.decomposition(x)
        trans, atm, A = self.fognet(torch.cat([x, lf], dim=1))
        streak = self.rainnet(torch.cat([x, hf], dim=1))
        streak = streak.repeat(1, 3, 1, 1)
        dehaze = (x - (1 - trans) * atm) / (trans + self.eps) - streak
        clean = self.relu(self.ref(dehaze, A))
        return streak, trans, atm, clean  # trans, atm, clean

    def forward_test(self, x, A, mode='A'):
        if mode=='run':
            lf, hf = self.decomposition(x)
            trans, atm, _ = self.fognet(torch.cat([x, lf], dim=1))
            streak = self.rainnet(torch.cat([x, hf], dim=1))
            streak = streak.repeat(1, 3, 1, 1)
            #dehaze = (x - 0.7*streak - (1 - trans) * atm) / (trans + self.eps)
            dehaze = (x - (1 - trans) * atm) / (trans + self.eps) - streak
            clean = self.relu(self.ref(dehaze, A))
            return streak, trans, atm, clean  # trans, atm, clean
        else:
            Amean = self.predict_atm(x)
            return Amean

    def predict_atm(self, x):
        _, c, h, w = x.size()
        lf, hf = self.decomposition(x)
        sum_A = torch.zeros(1,3,1,1, dtype=torch.float32).cuda()
        count = 0
        ph = 256
        pw = 256
        rows = int(np.ceil(float(h)/float(ph)))
        cols = int(np.ceil(float(w)/float(pw)))
        for i in range(rows):
            for j in range(cols):
                ratey = 1
                ratex = 1
                if (i+1)*256 > h:
                    dy = h - 256
                    ratey = h - i*256
                else:
                    dy = i * 256
                if (j+1) * 256 > w:
                    dx = w - 256
                    ratex = w - j*256
                else:
                    dx = j * 256
                _, _, A = self.fognet(torch.cat([x, lf], dim=1), dx, dy)
                sum_A = sum_A + A * ratey * ratex
                count+=1 * ratey * ratex
        return sum_A/count

    def decomposition(self, x):
        LF_list = []
        HF_list = []
        res = get_residue(x)
        res = res.repeat(1, 3, 1, 1)
        for radius in self.radiux:
            for eps in self.eps_list:
                self.gf = GuidedFilter(radius, eps)
                LF = self.gf(res, x)
                LF_list.append(LF)
                HF_list.append(x - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF


class FogNet(nn.Module):
    def __init__(self, input_nc):
        super(FogNet, self).__init__()
        self.atmconv1x1 = nn.Conv2d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
        self.atmnet = define_D(input_nc, 64, 'n_estimator', n_layers_D=5, norm='batch', use_sigmoid=True, gpu_ids=[])
        self.transnet = TransUNet(input_nc, 1)
        self.htanh = nn.Hardtanh(0, 1)
        self.relu1 = ReLU1()
        self.relu = nn.ReLU()

    def forward(self, x, dx=0, dy=0):
        _, c, h, w = x.size()

        A = self.relu(self.atmnet(self.atmconv1x1(x[:, :, dy:dy+256, dx:dx+256])))
        trans = self.relu(self.transnet(x))
        atm = A.repeat(1, 1, h, w)
        trans = trans.repeat(1, 3, 1, 1)
        return trans, atm, A


class TransUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(TransUNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.inc = inconv(in_channel, 64)
        self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.relu(self.outc(x))
        return x


class GlobalLocalUnet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(GlobalLocalUnet, self).__init__()
        self.inc = inconv(in_channel, 64)
        self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64
        self.down5 = down(512, 512)  # 32x32

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)
        self.outc = outconv(64, n_classes)
        # for global
        self.gdown5 = down(512, 512)  # 14x14  | 32x32  | 64x64
        self.gdown6 = down(512, 512)  # 7x7x512   | 16x16x512  | 32x32x512
        self.digest = nn.Conv2d(1024, 512, 1, 1, 0)
        self.l1 = nn.Linear(self.image_size / 64 * self.image_size / 64 * 512, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # local branch
        lx5 = self.down5(x5)

        # global branch
        gx5 = self.gdown5(x5)
        gx6 = self.gdown6(gx5)
        gx6vec = gx6.view(gx6.size(0), -1)
        gx7vec = self.relu(self.l1(gx6vec))
        gx7conv = gx7vec.view(gx7vec.size(0), -1, 1, 1).repeat(1, 1, self.image_size / 32, self.image_size / 32)

        # decoder
        u0 = torch.cat((lx5, gx7conv), dim=1)
        u0 = self.relu(self.digest(u0))
        x = self.up1(u0, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.relu(self.outc(x))
        return x


class RefUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(RefUNet, self).__init__()
        self.inc = inconv(in_channel, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024+32, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.Aup1 = nn.ConvTranspose2d(3,16,2,stride=2)
        self.Aup2 = nn.ConvTranspose2d(16,32,2,stride=2)
        # self.Aup3 = up(32,64,bilinear=False)

    def forward(self, x, A):
        input_img = x
        _, c, h, w = x.size()
        A1 = A.repeat(1, 1, h/64, w/64)
        A2 = self.Aup1(A1)  # 4x4
        A3 = self.Aup2(A2)  # 8x8 A3=16x16
        # A4 = self.Aup3(A3) # 16x16 A4=16x16
        x1 = self.inc(x)
        x2 = self.down1(x1)  # 256x256
        x3 = self.down2(x2)  # 128x128
        x4 = self.down3(x3)  # 64x64
        x5 = self.down4(x4)  # 32x32  x5=16x16
        x5 = torch.cat([x5, A3], dim=1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.relu(self.outc(x))
        #out = x + input_img
        return out


class DepthGuidedD(nn.Module):
    def __init__(self, nc):
        super(DepthGuidedD, self).__init__()
        self.conv1 = nn.Conv2d(nc, 8, 5,1,2)
        self.conv2 = nn.Conv2d(8, 16, 5,1,2)
        self.conv3 = nn.Conv2d(16, 64, 5,1,2)
        self.conv4 = nn.Conv2d(64, 128,5,1,2)
        self.conv5 = nn.Conv2d(128,128,5,1,2)
        self.conv6 = nn.Conv2d(128,128,5,1,2)
        self.convdepth = nn.Conv2d(128,1,5,1,2)
        self.conv7 = nn.Conv2d(128,64,5,4,1)
        self.conv8 = nn.Conv2d(64,32,5,4,1)
        self.conv9 = nn.Conv2d(32, 16, 5, 4, 1)
        self.fc = nn.Linear(32*16*16,1024)
        self.fc2 = nn.Linear(1024,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        depth = self.convdepth(x)
        x = self.relu(self.conv7(x*depth))
        x = self.relu(self.conv8(x))
        #x = self.relu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return depth, self.sigmoid(x)

# class RefUnet(nn.Module):
#     def __init__(self, in_channel):
#         super(RefUnet, self).__init__()
#         self.inc = inconv(in_channel, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 1024)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 32)
#         self.outc = outconv(64, n_classes)
#         self.leakyRelu = nn.LeakyReLU(0.1)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)


class RainNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RainNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.block1 = ResidualBlockUp(in_ch, 256)
        self.block2 = ResidualBlockStraight(256, 256)
        self.block3 = ResidualBlockStraight(256, 256)
        self.block4 = ResidualBlockDown(256, out_ch, last=True)
        # self.conv1010 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.conv1020 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.conv1030 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.conv1040 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        # self.upsample = F.upsample
        self.relu = nn.ReLU()
        self.reset_params()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # shape_out = (x.size()[2], x.size()[3])
        # x101 = F.avg_pool2d(x, 32)
        # x102 = F.avg_pool2d(x, 16)
        # x103 = F.avg_pool2d(x, 8)
        # x104 = F.avg_pool2d(x, 4)
        #
        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out, mode='bilinear')
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out, mode='bilinear')
        # x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out, mode='bilinear')
        # x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out, mode='bilinear')
        # p = torch.cat([x1010,x1020,x1030,x1040,x], dim=1)
        x = self.block4(x)
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


# -------------------------------------------------------------------------- #
#  Sub-Networks
# -------------------------------------------------------------------------- #
class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, dil=1, middle_planes=32):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=dil, dilation=dil)
        self.layer1 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer2 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer3 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer4 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.conv_out = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        return x


class ResidualBlockStraight(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=False):
        super(ResidualBlockStraight, self).__init__()
        assert (in_channels == out_channels)
        self.conv1 = res_conv(in_channels, out_channels / 4, dil=dilation)
        self.conv2 = res_conv(out_channels / 4, out_channels / 4)
        self.conv3 = res_conv(out_channels / 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=True):
        super(ResidualBlockDown, self).__init__()
        self.conv1 = res_conv(in_channels, in_channels, dil=dilation)
        self.conv2 = res_conv(in_channels, in_channels / 2)
        self.conv3 = res_conv(in_channels / 2, in_channels / 4)
        self.conv_out = nn.Conv2d(in_channels + in_channels / 4, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = torch.cat((out, residual), dim=1)
        out = self.relu(out)
        out = self.conv_out(out)
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super(ResidualBlockUp, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels / 4, kernel_size=5, stride=1, padding=2)
        self.conv1 = res_conv(out_channels / 4, out_channels / 4)
        self.conv2 = res_conv(out_channels / 4, out_channels / 4)
        self.conv3 = res_conv(out_channels / 4, out_channels)
        self.conv_in = nn.Conv2d(out_channels / 4, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        x = self.relu(self.conv0(x))
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        residual = self.conv_in(residual)
        out = out + residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


# ---------------------------------------------------------------------------- #
# Sub-Modules
# ---------------------------------------------------------------------------- #


class ReLU1(Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1, self).__init__(0, 1, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


def res_conv(in_ch, out_ch, k=3, s=1, dil=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, dilation=dil, padding=1, bias=bias)


class SimpUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(SimpUnet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            # bn = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            # bn = True if i> 0 and i < depth-2 else False
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class RainUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        super(RainUnet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.ConvRefine = nn.Conv2d(outs, 32, 3, 1, 1)
        self.RefineClean1 = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)
        self.RefineClean2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.RefineClean3 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.LRelu = nn.LeakyReLU(0.2, inplace=True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        input_img = x
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.

        x9 = self.LRelu((self.ConvRefine(x)))

        residual = F.tanh(self.RefineClean3(x9))
        clean1 = self.LRelu(self.RefineClean1(residual))
        clean2 = F.tanh(self.RefineClean2(clean1))
        return clean2


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
#  Network Architecture
# -------------------------------------------------------------------------- #
def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, dropout_rate=0.5):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x

        if self.pooling:
            x = self.pool(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', dropout_rate=0.5):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.dropout_rate = dropout_rate

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        # layer 1
        x = F.relu(self.conv1(x))
        # layer 2
        x = F.relu(self.conv2(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x

# ============================================================================
