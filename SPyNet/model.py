import torch
import torch.nn as nn
import math

import numpy
import time

from PIL import Image

PRETRAIN_MODEL = 'sintel-final'
IMAGE1_PATH = 'image1.png'
IMAGE2_PATH = 'image2.png'
FLOW_SAVE_PATH = 'out.flo'

class SPyNet(nn.Module):
    def __init__(self):
        super(SPyNet, self).__init__()

        class Preprocess(nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, x):
                B = (x[:, 0, :, :] - 0.406) / 0.225
                G = (x[:, 1, :, :] - 0.456) / 0.224
                R = (x[:, 2, :, :] - 0.485) / 0.229

                return torch.cat([ R, G, B ], 1)

        class Basic(nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.netBasic = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, x):
                return self.netBasic(x)

        self.netPreprocess = Preprocess()
        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
        self.load_state_dict(
            { strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('run.py', 'network-' + PRETRAIN_MODEL + '.pytorch')).items() })

    def forward(self, tensor1, tensor2):
        flow = []

        tensor1 = [ self.netPreprocess(tensor1) ]
        tensor2 = [ self.netPreprocess(tensor2) ]

        for intLevel in range(5):
            if tensor1[0].shape[2] > 32 or tensor1[0].shape[3] > 32:
                tensor1.insert(0, torch.nn.functional.avg_pool2d(input=tensor1[0], kernel_size=2, stride=2, count_include_pad=False))
                tensor2.insert(0, torch.nn.functional.avg_pool2d(input=tensor2[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = tensor1[0].new_zeros([ tensor1[0].shape[0], 2, int(math.floor(tensor1[0].shape[2] / 2.0)), int(math.floor(tensor1[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tensor1)):
            tensor_upsampled = torch.nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensor_upsampled.shape[2] != tensor1[intLevel].shape[2]: tensor_upsampled = torch.nn.functional.pad(input=tensor_upsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tensor_upsampled.shape[3] != tensor1[intLevel].shape[3]: tensor_upsampled = torch.nn.functional.pad(input=tensor_upsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            flow = self.netBasic[intLevel](torch.cat([ tensor1[intLevel], backwarp(tensor=tensor2[intLevel], flow=tensor_upsampled), tensor_upsampled ], 1)) + tensor_upsampled

        return flow

backwarp_tenGrid = {}

def backwarp(tensor, flow):
    if str(flow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, flow.shape[3]).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, flow.shape[2]).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])

        backwarp_tenGrid[str(flow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()

    flow = torch.cat([ flow[:, 0:1, :, :] / ((tensor.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] / ((tensor.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensor, grid=(backwarp_tenGrid[str(flow.size())] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

def estimate(tenFirst, tenSecond):
    netNetwork = SPyNet().cuda().eval()

    assert(tenFirst.shape[1] == tenSecond.shape[1])
    assert(tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    assert(intWidth == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 240) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()

def main():
    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(IMAGE1_PATH))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(IMAGE2_PATH))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    print(tenFirst.shape)
    start_time = time.time()
    tenOutput = estimate(tenFirst, tenSecond)
    print(time.time() - start_time)

    objOutput = open(FLOW_SAVE_PATH, 'wb')

    numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

    objOutput.close()

main()