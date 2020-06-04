# Largely adapted from https://github.com/gengshan-y/VCN/blob/master/visualize.ipynb

import torch
import torch.nn as nn

import cv2
import numpy as np
import imageio
from matplotlib import pyplot as plt

from models.VCN import VCN
from utils.flowlib import read_flow, flow_to_image


# Hyperparameters
MAXDISP = 256                       # Maximum disparity to search over (along each direction)
FAC=1                               # Shape of search window
MODEL_PATH = 'finetune_67999.tar'   

# Put the flow vectors on top of image
def point_vec(img,flow):
    meshgrid = np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    dispimg = cv2.resize(img, None,fx=4,fy=4)
    colorflow = flow_to_image(flow).astype(int)
    for i in range(img.shape[1]): # x 
        for j in range(img.shape[0]): # y
            if flow[j,i,2] != 1: continue
            if j%10!=0 or i%10!=0: continue
            xend = int((meshgrid[0][j,i]+flow[j,i,0])*4)
            yend = int((meshgrid[1][j,i]+flow[j,i,1])*4)
            leng = np.linalg.norm(flow[j,i,:2])
            if leng<1:continue
            dispimg = cv2.arrowedLine(dispimg, (meshgrid[0][j,i]*4,meshgrid[1][j,i]*4),\
                                      (xend,yend),
                                      (int(colorflow[j,i,2]),int(colorflow[j,i,1]),int(colorflow[j,i,0])),5,tipLength=8/leng,line_type=cv2.LINE_AA)
    return dispimg

def showImage(image, save_name):
    plt.figure(figsize=(25,25))
    plt.imshow(image)
    plt.imsave(save_name, image)
    plt.show()

def calOpt(height=240, width=320, maxdisp=256, fac=1, modelpath='finetune_67999.tar'):
    # Calculate model hyperparameters
    # Resize to 64X
    maxh = height
    maxw = width
    max_h = int(maxh // 64 * 64)            # Basically this is performing an integer division and modulo operation
    max_w = int(maxw // 64 * 64)            # if modulo is not zero, then round it up
    if max_h < maxh:                        # The rounded-up integer is multiplied by 64x
        max_h += 64
    if max_w < maxw: 
        max_w += 64

    # load model
    model = VCN([1, max_w, max_h], md=[int(4*(maxdisp/256)),4,4,4,4], fac=fac)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    # load weights
    pretrained_dict = torch.load(modelpath)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    model.eval()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    # Load image and Resize
    # Note that the images are loaded as [H,W,C] i.e. [H,W,3]
    imgL_o = imageio.imread('image1.png')[:,:,:3]        # In some cases, image files include alpha channel (the 4th channel)
    imgR_o = imageio.imread('image2.png')[:,:,:3]        # Only get the RGB channels (1st 3 channels)
    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))

    # For gray input images
    # The model expects RGB images, in other words, 3 channels
    # This repeats H*W spatial values over the channel layer [H,W,1] -> [H,W,3]
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

    # Flip channel, subtract mean
    # The model expects inputs of format [C,H,W] instead of [H,W,C]
    imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    # Image to Torch tensor
    imgL = torch.FloatTensor(imgL).cuda()       
    imgR = torch.FloatTensor(imgR).cuda()       

    # Forward
    with torch.no_grad():
        imgLR = torch.cat([imgL,imgR],0)
        rts = model(imgLR)
        pred_disp, entropy = rts

    # Upsampling
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()                                 # Remove batch dimension, torch tensor to numpy ndarray
    pred_disp = cv2.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0])) # Resize to the original size, and transpose from [C,H,W] -> [H,W,C]
    pred_disp[:,:,0] *= input_size[1] / max_w
    pred_disp[:,:,1] *= input_size[0] / max_h
    flow = np.ones([pred_disp.shape[0],pred_disp.shape[1],3])
    flow[:,:,:2] = pred_disp
    entropy = torch.squeeze(entropy).data.cpu().numpy()
    entropy = cv2.resize(entropy, (input_size[1], input_size[0]))

    # Show results
    showImage( flow_to_image(flow), "flow_to_image.png")
    showImage( point_vec(imgL_o,flow)[:,:,::-1], "vector_on_image.png")
    showImage( entropy, "entropy.png")
    
calOpt()