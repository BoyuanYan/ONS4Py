import torch
import torch.nn as nn
import time

global lastlog
lastlog = time.time()


def mylog(content, cls="DEBUG"):
    now = time.time()
    global lastlog
    print("{cls}: {content}, interval:{interval}, from device {device}".format(
        cls=cls, content=content, interval=now - lastlog, device=torch.cuda.current_device()))
    lastlog = now


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, apply_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.apply_relu = apply_relu
        if self.apply_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.apply_relu:
            x = self.relu(x)
        return x


class Inception(nn.Module):

    def __init__(self, in_planes, out_planes, branches, reduce=False):
        super(Inception, self).__init__()
        assert(type(branches) == type([]) and len(branches)==3)
        stride = 1
        if reduce:
            stride = 2
        self.projection = (in_planes != out_planes)
        if self.projection:
            self.branch0 = BasicConv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, apply_relu=False)

        self.branch1 = BasicConv2d(in_planes, branches[0], kernel_size=1, stride=stride)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, branches[1], kernel_size=1, stride=1),
            BasicConv2d(branches[1], branches[1],
                        kernel_size=3, stride=stride, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes, branches[2], kernel_size=1, stride=1),
            BasicConv2d(branches[2], branches[2], kernel_size=3, stride=1, padding=1),
            BasicConv2d(branches[2], branches[2],
                        kernel_size=3, stride=stride, padding=1),
        )

        self.conv = BasicConv2d(sum(branches), out_planes, kernel_size=1,
                                stride=1, apply_relu=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        if self.projection:
            x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), 1)
        out = self.bn(self.conv(out))
        out = self.relu(out + x0)
        return out


class IR18(nn.Module):

    def __init__(self, num_classes=2):
        super(IR18, self).__init__()
        # TODO preprocess 是新增的处理。因为输入是3x448x448的数据
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            BasicConv2d(3, 24, kernel_size=3, stride=2, padding=1),
            BasicConv2d(24, 12, kernel_size=3, stride=1, padding=1),
            BasicConv2d(12, 96, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Inception(96, 384, [24, 48, 48]),
            Inception(384, 384, [24, 48, 48]),
        )
        self.stage2 = nn.Sequential(
            Inception(384, 96, [24, 48, 48], reduce=True),
            Inception(96, 96, [24, 48, 48]),
        )
        self.stage3 = nn.Sequential(
            Inception(96, 384, [48, 96, 96], reduce=True),
            Inception(384, 384, [96, 192, 192]),
        )
        self.stage4 = nn.Sequential(
            nn.AvgPool2d(3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=1, ceil_mode=True),
        )
        self.stage5 = nn.Sequential(
            nn.Linear(384, 196),
            nn.ReLU(inplace=True),
            nn.Linear(196, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.stage5(x)
        return x


def test():
    model = IR18()
    model.eval()
    inputs = torch.autograd.Variable(torch.zeros(2, 3, 224, 224))
    mylog("Testing")
    out4 = model(inputs)
    print(out4)


if __name__ == '__main__':
    test()
