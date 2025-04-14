import numpy as np

def _k(BL,I):

  k_value=[]
  for i in range(3):
    channel=I[:,:,i]
    B_c=BL[i]
    value=np.max((channel-B_c))
    k_value.append(value)

  return max(k_value)

def d_o(BL, I):
    max_diffs = []
    for c in range(3):
        max_diffs.append(np.max(np.abs(I[:,:,c] - BL[c])))
    k = np.argmax(max_diffs)
    B_k = max_diffs[k]
    d0 = (B_k / max(BL[k], 1 - BL[k]))
    return d0

