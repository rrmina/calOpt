import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import datasets
from utils import flow_viz
from raft import RAFT

import time

DEVICE = 'cuda'

def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    pad_ht1 = [pad_ht//2, pad_ht-pad_ht//2]
    pad_wd1 = [pad_wd//2, pad_wd-pad_wd//2]

    img = F.pad(img, pad_wd1 + pad_ht1, mode='replicate')
    return img

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return pad8(img[None]).to(DEVICE)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # BGR TO RGB
    tensor = torch.from_numpy(img).permute(2,0,1).float()       # [H,W,C] to [C,H,W]
    tensor = pad8(tensor[None]).to(DEVICE)  
    return tensor

def postprocess(flow,w,h):
    flow = flow[-1][0]
    flow = flow.permute(1,2,0).cpu().numpy()                    # [C,H,W] to [H,W,C]
    flow_image = flow_viz.flow_to_image(flow)               
    flow_image = cv2.resize(flow_image, (w,h))  
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_RGB2BGR)    # RGB to BGR
    return flow_image

def display(image1, image2, flow):
    image1 = image1.permute(1, 2, 0).cpu().numpy() / 255.0
    image2 = image2.permute(1, 2, 0).cpu().numpy() / 255.0

    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[1], image1.shape[0]))

    cv2.imshow('image1', image1[..., ::-1])
    cv2.imshow('image2', image2[..., ::-1])
    cv2.imshow('flow', flow_image[..., ::-1])
    cv2.waitKey()

def demo(args):
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))

    model.to(DEVICE)
    model.eval()

    arr = []

    
    with torch.no_grad():

        cap = cv2.VideoCapture('video.mp4')
        _, left_frame = cap.read()
        h,w,_ = left_frame.shape
        left_tensor = preprocess(left_frame)

        while (1):
            _, right_frame = cap.read()
            right_tensor = preprocess(right_frame)

            start1 = time.time()
            flow_predictions = model(left_tensor, right_tensor, iters=args.iters, upsample=False)
            print(time.time() - start1)

            flow_image = postprocess(flow_predictions,w,h)
            cv2.imshow('frame', flow_image)


            k = cv2.waitKey(25)
            if (k == 27):
                break

            left_tensor = right_tensor.clone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    args = parser.parse_args()
    demo(args)