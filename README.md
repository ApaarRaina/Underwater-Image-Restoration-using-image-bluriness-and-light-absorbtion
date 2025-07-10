# Underwater Image Restoration using Image Blurriness and Light Absorption (IBLA)

This repository contains a **Python implementation** of the method described in the IEEE paper:

**"Underwater Image Restoration Based on Image Blurriness and Light Absorption"**  
by Yan-Tsung Peng, Pamela C. Cosman.  
IEEE Transactions on Image Processing, 2017.  
[Paper Link (IEEE Xplore)](https://ieeexplore.ieee.org/document/7840002)

---

## Overview

Underwater images often suffer from degradation due to scattering and light absorption. This implementation is based on the IBLA (Image Blurriness and Light Absorption) method, which improves visibility and contrast in underwater images using:

- **Local image blurriness maps**  
- **Light attenuation priors**  
- **Quad-tree-based adaptive background light estimation**  
- **Depth map estimation**  
- **Transmission map refinement**

This project is a full reimplementation of the IBLA method in Python, inspired by the original MATLAB code by Jerry Peng.

---
## How to Run

1. **Clone the repository**

```bash
git clone [https://github.com/ApaarRaina/Underwater-Image-Restoration-using-image-bluriness-and-light-absorbtion.git]
cd underwater-image-restoration

python dip_proj.py
```
## Results

Below are the restored images(left) by the algorithm 

<div style="display: flex; justify-content: space-between;">
  <img src="Enhanced Image-2.png" alt="Generated MNIST Digits" width="300"/>
  <img src="Enhanced Image-1.png" alt="Generated MNIST Digits" width="300"/>
</div>

