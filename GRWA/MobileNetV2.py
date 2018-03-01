import torch.nn as nn
import torch


class MobileNetV2(nn.Module):
    """
    mobilenet V2。原文可见：https://arxiv.org/abs/1801.04381

    下述结构即MobileNet v2的结构，其中channels表示扩张倍数，c表示输出channels个数，n表示重复次数，s表示stride

    |     name     |    Input   |  Operator  | t |  c  | n | s |
    | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- |
    |     conv_1   | 224x224 x3 | conv2d 3x3 | - | 32  | 1 | 2 |
    | bottleneck_1 | 112x112x32 | bottleneck | 1 | 16  | 1 | 1 |
    | bottleneck_2 | 112x112x16 | bottleneck | 6 | 24  | 2 | 2 |
    | bottleneck_3 | 56 x56 x24 | bottleneck | 6 | 32  | 3 | 2 |
    | bottleneck_4 | 28 x28 x32 | bottleneck | 6 | 64  | 4 | 2 |
    | bottleneck_5 | 14 x14 x64 | bottleneck | 6 | 96  | 3 | 1 |
    | bottleneck_6 | 14 x14 x96 | bottleneck | 6 | 160 | 3 | 2 |
    | bottleneck_7 | 7  x7 x160 | bottleneck | 6 | 320 | 1 | 1 |
    |    conv_2    | 7  x7 x320 | conv2d 1x1 | - | 1280| 1 | 1 |
    |    avgpool   | 7  x7 x1280| avgpool 7x7| - |  -  | 1 | - |
    |      fc      | 1  x1 xk   |   Linear   | - |  k  | - |   |
    """

    def __init__(self, in_channels: int=3, num_classes: int=1000, t: int=6):
        super(MobileNetV2, self).__init__()
        #          c1, b1, b2, b3, b4, b5, b6,  b7,  c2
        out_chs = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        strides = [ 2,  1,  2,  2,  2,  1,   2,   1,    1]
        r_times = [ 1,  1,  2,  3,  4,  3,   3,   1,    1]
        factors = [-1,  1,  t,  t,  t,  t,   t,   t,   -1]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_chs[0], kernel_size=3,
                      stride=strides[0], padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chs[0]),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential()
        for i in range(7):
            name = 'bottleneck_' + str(i+1)
            bnk = StackBottleneck(in_features=out_chs[i], out_features=out_chs[i+1],
                                  first_stride=strides[i+1], factor_t=factors[i+1],
                                  repeated_times=r_times[i+1], name=name)
            self.bottleneck.add_module(name, bnk)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs[7], out_channels=out_chs[8], kernel_size=1,
                      stride=strides[8], padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_chs[8]),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, ceil_mode=True)

        self.fc = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bottleneck(x)
        x = self.conv_2(x)
        x = self.avgpool(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    """
    mobilenet v2中定义的bottleneck。
    """

    def __init__(self, in_features: int, out_features: int, stride: int, factor_t: int):
        """

        :param in_features:
        :param out_features:
        :param stride:
        :param factor_t:
        """
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.use_res = in_features == out_features
        middle_features = in_features * factor_t
        self.stem = nn.Sequential(
            # pointwise conv
            nn.Conv2d(in_channels=in_features, out_channels=middle_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=middle_features),
            nn.ReLU(inplace=True),
            # depthwise conv
            nn.Conv2d(in_channels=middle_features, out_channels=middle_features, kernel_size=3,
                      stride=stride, padding=1, groups=middle_features, bias=False),
            nn.BatchNorm2d(num_features=middle_features),
            nn.ReLU(inplace=True),
            ## pointwise conv
            nn.Conv2d(in_channels=middle_features, out_channels=out_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_features)
            # 最后一层无ReLU。
        )

    def forward(self, x):
        y = self.stem(x)
        if self.stride is 1 and self.use_res:
            y = y + x
        return y


class StackBottleneck(nn.Module):
    """
    mobilenet v2中bottlenecks的叠加
    """

    def __init__(self, in_features: int, out_features: int, first_stride: int,
                 factor_t: int, repeated_times: int, name: str):
        """

        :param in_features:
        :param out_features:
        :param first_stride:
        :param factor_t:
        :param repeated_times:
        """
        super(StackBottleneck, self).__init__()

        self.model = nn.Sequential(
            Bottleneck(in_features=in_features, out_features=out_features, stride=first_stride,
                       factor_t=factor_t)
        )

        for i in range(repeated_times-1):
            module = Bottleneck(in_features=out_features, out_features=out_features, stride=1,
                                factor_t=factor_t)
            self.model.add_module(name=name+'_'+str(i+1), module=module)

    def forward(self, x):
        y = self.model(x)
        return y


def test():
    model = MobileNetV2()
    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    y = model(x)
    print(y.size())