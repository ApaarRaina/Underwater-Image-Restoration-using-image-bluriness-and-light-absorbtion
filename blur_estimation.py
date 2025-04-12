import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_fill_holes


def blur_estimate(I_g,img_normalised):
    P_init = np.zeros((I_g.shape[0], I_g.shape[1]))

    for r in [9, 17, 33, 65]:
        G_i = cv.GaussianBlur(I_g, (r, r), 0)
        P_init += np.abs(I_g - G_i)

    P_init //= 4

    P_init = P_init.astype(np.float32) / 255.0

    # calculating rough bluriness map
    k_size = 7
    kernel = np.ones((k_size, k_size))

    P_r = maximum_filter(P_init, size=(k_size, k_size))

    # calculating P_blr
    kernel = (7, 7)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)

    # Step 1: Dilate the grayscale roughBlurMap
    dilated = cv.dilate(P_r, kernel)

    # Step 2: Fill holes (automatically uses 8-connectivity)
    # We normalize it temporarily for binary fill
    filled = binary_fill_holes(dilated.astype(bool))

    # Step 3: Mask the original dilated map
    P_blr = dilated * filled.astype(np.float32)
    C_r = P_blr
    radius = 7

    if img_normalised.shape[0] * img_normalised.shape[1] < 180000:
        lambda_val = 10e-6
    else:
        lambda_val = 10e-3

    P_blr = cv.ximgproc.guidedFilter(guide=img_normalised, src=P_blr, radius=radius, eps=lambda_val)

    return P_blr,C_r
