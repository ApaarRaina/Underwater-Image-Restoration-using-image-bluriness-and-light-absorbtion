import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from Backgroundlight_estimation import quad_tree_LB,quad_tree_LV,S,BL_estimate,get_top_pixels




#depth estimation

def F_s(V):
  return (V-np.min(V))/(np.max(V)-np.min(V))


def depth_mip(I_r,I_g,I_b):
  pool=nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
  red_tensor=torch.tensor(I_r,dtype=torch.float32)
  red_tensor=red_tensor.unsqueeze(0).unsqueeze(0)


  green_tensor=torch.tensor(I_g,dtype=torch.float32)
  green_tensor=green_tensor.unsqueeze(0).unsqueeze(0)

  blue_tensor=torch.tensor(I_b,dtype=torch.float32)
  blue_tensor=blue_tensor.unsqueeze(0).unsqueeze(0)

  max_r=pool(red_tensor)
  max_g=pool(green_tensor)
  max_b=pool(blue_tensor)

  max_gb=torch.maximum(max_g,max_b)

  depth=max_r-max_gb

  depth=depth.squeeze(0).squeeze(0).numpy()

  return depth


def depth_estimation(map_1,map_2,map_3,BL,red):

   O_a=S(np.mean(BL),0.5)
   O_b=S(np.mean(red),0.1)

   map=O_b*(O_a*map_2 + (1-O_a)*map_1) + (1-O_b)*map_3

   return map