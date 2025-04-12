import math
import numpy as np

e_s=math.pow(2.0,-10)
e_n=0.1

def S(v,a):
  return 1/(1+np.exp(-32*(a-v)))

def get_top_pixels(P_blr):
  thresh=np.percentile(P_blr,99.9)
  mask=P_blr>=thresh
  return mask

def quad_tree_LB(I_g,P_blr,e_s):

    size_I_g=I_g.shape[0]*I_g.shape[1]
    size_I_c=I_g.shape[0]*I_g.shape[1]
    I_c=P_blr
    start_row=0
    start_col=0
    end_row=I_g.shape[0]
    end_col=I_g.shape[1]

    while((size_I_c/size_I_g)>e_s):
      del_width=I_c.shape[0]//2
      del_height=I_c.shape[1]//2

      block_1=I_c[0:del_width,0:del_height]
      block_2=I_c[del_width:del_width+del_width,0:del_height]
      block_3=I_c[0:del_width,del_height:del_height+del_height]
      block_4=I_c[del_width:del_width+del_width,del_height:del_height+del_height]

      l_block=[block_1,block_2,block_3,block_4]


      mean_1=np.mean(block_1)
      mean_2=np.mean(block_2)
      mean_3=np.mean(block_3)
      mean_4=np.mean(block_4)

      l=[mean_1,mean_2,mean_3,mean_4]

      index=l.index(max(l))

      if index == 0:  # top-left
            pass
      elif index == 1:  # bottom-left
            start_row += del_height
      elif index == 2:  # top-right
            start_col += del_width
      elif index == 3:  # bottom-right
            start_row += del_height
            start_col += del_width


      I_c=l_block[index]
      size_I_c=I_c.shape[0]*I_c.shape[1]


    end_row = start_row + I_c.shape[0]
    end_col = start_col + I_c.shape[1]

    return start_row, end_row, start_col, end_col

def quad_tree_LV(I_g,e_s):

    size_I_g=I_g.shape[0]*I_g.shape[1]
    size_I_c=I_g.shape[0]*I_g.shape[1]
    I_c=I_g
    start_row=0
    start_col=0
    end_row=I_g.shape[0]
    end_col=I_g.shape[1]

    while((size_I_c/size_I_g)>e_s):
      del_width=I_c.shape[0]//2
      del_height=I_c.shape[1]//2

      block_1=I_c[0:del_width,0:del_height]
      block_2=I_c[del_width:del_width+del_width,0:del_height]
      block_3=I_c[0:del_width,del_height:del_height+del_height]
      block_4=I_c[del_width:del_width+del_width,del_height:del_height+del_height]

      l_block=[block_1,block_2,block_3,block_4]


      var_1=np.var(block_1)
      var_2=np.var(block_2)
      var_3=np.var(block_3)
      var_4=np.var(block_4)

      l=[var_1,var_2,var_3,var_4]

      index=l.index(min(l))

      if index == 0:  # top-left
            pass
      elif index == 1:  # bottom-left
            start_row += del_height
      elif index == 2:  # top-right
            start_col += del_width
      elif index == 3:  # bottom-right
            start_row += del_height
            start_col += del_width


      I_c=l_block[index]
      size_I_c=I_c.shape[0]*I_c.shape[1]


    end_row = start_row + I_c.shape[0]
    end_col = start_col + I_c.shape[1]

    return start_row, end_row, start_col, end_col


def BL_estimate(BL_1,BL_2,BL_3,I_r,I_gr,I_b):

  I=[I_r,I_gr,I_b]

  final_BL=[]
  for i in range(len(I)):
      I_c=I[i]
      n=np.sum(I_c>0.5)
      a=S(n/(I_c.shape[0]*I_c.shape[1]),e_n)

      BL_c_max=max(BL_1[i],BL_2[i],BL_3[i])
      BL_c_min=min(BL_1[i],BL_2[i],BL_3[i])

      BL_c=a*BL_c_max + (1-a)*BL_c_min

      final_BL.append(BL_c.item())




  return final_BL



