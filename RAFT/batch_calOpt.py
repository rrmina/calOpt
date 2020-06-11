import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

from utils import flow_viz
from raft import RAFT

import time

DEVICE = 'cuda'
IMAGE_FOLDER_PATH = "folder_containing_framesfolder/"
BATCH_SIZE = 75

class OpticalFlowFolder(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder()
    Reference: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """

    def __init__(self, root, transform):
        super(OpticalFlowFolder, self).__init__(root=root, transform=transform)

        frame_folder = os.path.join(root, 'frames')

        FRAME_BASE_FILE_NAME = "frame"
        FRAME_BASE_FILE_tYPE = ".jpg"

        base_name_len = len(FRAME_BASE_FILE_NAME)
        filetype_len = len(FRAME_BASE_FILE_tYPE)

        # Natural sorting of frames. Unfortunately, Python doesn't have a native support for natural sorting.
        sorted_framename = sorted(
            os.listdir(frame_folder),
            key=lambda x: int(x[base_name_len:-filetype_len])
        )

        # Tuples of (image_path, label)
        # Label is set default to 0
        self.imgs = [(os.path.join(frame_folder, framename), 0) for framename in sorted_framename]

    # Override the __len__ method. This method is called to determine the upper limit of __getitem__ indeces
    # Note that given N frames, there are (N-1) optical flow input pairs
    def __len__(self):
        return len(self.imgs)-1

    # Override the __getitem__ method. This is the method dataloader calls
    def __getitem__(self, index):
        # Shift indeces of the images
        index1 = index
        index2 = index + 1

        # This is what ImageFolder normally returns 
        original_tuple = super(OpticalFlowFolder, self).__getitem__(index1)
        shifted_tuple = super(OpticalFlowFolder, self).__getitem__(index2)

        # Make a new tuple that includes original batch, and the shifted batch
        tuple_return = (original_tuple[0], shifted_tuple[0])
        return tuple_return

def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    pad_ht1 = [pad_ht//2, pad_ht-pad_ht//2]
    pad_wd1 = [pad_wd//2, pad_wd-pad_wd//2]

    img = F.pad(img, pad_wd1 + pad_ht1, mode='replicate')
    return img

def batch_calOpt(args):
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))
    
    model.to(DEVICE)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ImageFolder and Loader
    image_dataset = OpticalFlowFolder( IMAGE_FOLDER_PATH, transform=transform )
    image_loader = torch.utils.data.DataLoader( image_dataset, batch_size=BATCH_SIZE )

    start = time.time()
    with torch.no_grad():
        for left, right in image_loader:
            # Most of the time, this preprocessing is not needed
            # Especially if the video dimensions are multiple of 8s
            _, _, h, w = left.shape
            if ((h % 8 != 0) or (w % 8 != 0)):
                left = pad8(left)
                right = pad8(right)

            # Forward
            flow_predictions = model(left, right, iters=args.iters, upsample=False)

    print("Time Elapsed: ", time.time() - start)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    args = parser.parse_args()
    batch_calOpt(args)
