import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2.ximgproc as xip
from blur_estimation import blur_estimate
from Backgroundlight_estimation import quad_tree_LB,quad_tree_LV,S,BL_estimate,get_top_pixels
from depth_estimation import F_s,depth_mip,depth_estimation
from TM_estimation import d_o,_k
import math

img=cv.imread('galdran1_input.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

print(img.shape)

img_normalised = img.astype(np.float32) / 255.0



"""---------------------------------# Part A Estimation of bluriness map P_blr----------------------------"""

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
I_g = img_gray

radius = 7

if img_normalised.shape[0] * img_normalised.shape[1] < 180000:
    lambda_val = 10e-6
else:
    lambda_val = 10e-3


P_blr,C_r=blur_estimate(I_g,img_normalised)


"""-----------------------------# Part B  BL (Background Light) Estimation-------------------------------"""

#background light estimate

e_s=math.pow(2.0,-10)
e_n=0.1

#apply mask to get the first candidate
mask=get_top_pixels(P_blr)
#BL_1=img_normalised*mask

red=img_normalised[:,:,0]
green=img_normalised[:,:,1]
blue=img_normalised[:,:,2]


# did this so that the 0 value that we would get after applying the mask do not dilute the mean
BL_1_red = np.mean(red[mask == 1])
BL_1_green = np.mean(green[mask == 1])
BL_1_blue = np.mean(blue[mask == 1])

BL_1=np.array([BL_1_red,BL_1_green,BL_1_blue])


start_row,end_row,start_col,end_col=quad_tree_LV(I_g,e_s)

red=img_normalised[start_row:end_row,start_col:end_col,0]
green=img_normalised[start_row:end_row,start_col:end_col,1]
blue=img_normalised[start_row:end_row,start_col:end_col,2]



BL_2_red=np.mean(red)
BL_2_green=np.mean(green)
BL_2_blue=np.mean(blue)

BL_2=np.array([BL_2_red,BL_2_green,BL_2_blue])


start_row,end_row,start_col,end_col=quad_tree_LB(I_g,P_blr,e_s)

red=img_normalised[start_row:end_row,start_col:end_col,0]
green=img_normalised[start_row:end_row,start_col:end_col,1]
blue=img_normalised[start_row:end_row,start_col:end_col,2]

BL_3_red=np.mean(red)
BL_3_green=np.mean(green)
BL_3_blue=np.mean(blue)

BL_3=np.array([BL_3_red,BL_3_green,BL_3_blue])


red=img_normalised[:,:,0]
green=img_normalised[:,:,1]
blue=img_normalised[:,:,2]

BL=BL_estimate(BL_1,BL_2,BL_3,red,green,blue)


print(BL)

"""---------------------------------# Part C Depth Estimation----------------------------------------"""


red=img_normalised[:,:,0]
green=img_normalised[:,:,1]
blue=img_normalised[:,:,2]

pool=nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
red_tensor=torch.tensor(red,dtype=torch.float32)
red_tensor=red_tensor.unsqueeze(0).unsqueeze(0)
R=pool(red_tensor)
R=R.squeeze(0).squeeze(0).numpy()

map_1=1-F_s(R)

map_2=F_s(depth_mip(red,green,blue))

map_3=F_s(C_r)

map=depth_estimation(map_1,map_2,map_3,BL,red)

map = xip.guidedFilter(guide=img_normalised, src=map, radius=radius, eps=lambda_val)

"""---------------------# Part D TM Estimation and Scene Radiance Recovery------------------------"""

#TM estimation and Scene Radiance Recovery

d0=d_o(BL,img_normalised)

d_f=8*(map+d0)
d_f = np.clip(d_f, 0.1, 8)  # Avoid exploding exp
beta_r=1/7
m=-0.00113
i=1.62517

h_g=540
h_r=620
h_b=450

beta_g=(BL[0]*(m*h_g+i))/(BL[1]*(m*h_r+i))
beta_b=(BL[0]*(m*h_b+i))/(BL[2]*(m*h_r+i))

t_r=np.exp(-beta_r*d_f)
t_b=(t_r**beta_b)
t_g=(t_r**beta_g)
tm=[t_r,t_b,t_g]

J = np.zeros_like(img_normalised)
t0 = 0.1
for i in range(3):
    t_c = np.maximum(tm[i], t0)
    J[:,:,i] = (img_normalised[:,:,i] - BL[i]) /t_c + BL[i]



J = np.clip(J, 0, 1)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow((J * 255).astype(np.uint8))
plt.title('Restored Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_normalised)
plt.title('Original Image')
plt.axis('off')
plt.show()

new_img=J
err=img_normalised-new_img
print(np.mean(err))