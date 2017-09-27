import numpy as npy
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import sys
import math

def save_hue(scan_num,try_num):

    dir_name = './recon_result/S'+scan_num+'/'+try_num+'/recon_data'
    prb_file_name = 'recon_'+scan_num+'_'+try_num+'_probe_ave_rp'
    obj_file_name = 'recon_'+scan_num+'_'+try_num+'_object_ave_rp'

    data_prb = npy.load(dir_name+'/'+prb_file_name+'.npy')
    
    H_mean = npy.mean(npy.angle(data_prb))
    H = ((npy.angle(data_prb) - H_mean + math.pi) * 180. / math.pi)/360.

    V = npy.abs(data_prb) / npy.max(npy.abs(data_prb))
    S = npy.ones_like(V)

    HSV = npy.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    plt.figure()
    plt.imshow(npy.fliplr(npy.rot90(npy.flipud(RGB))),interpolation='none')
    plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+prb_file_name+'_hsv.png')

    #+++++++++++++++++++++++++++++++++

    data_obj = npy.load(dir_name+'/'+obj_file_name+'.npy')

    H_mean = npy.mean(npy.angle(data_obj))
    H = ((npy.angle(data_obj) - H_mean + math.pi) * 180. / math.pi)/360.

    V = npy.abs(data_obj) / npy.max(npy.abs(data_obj))
    S = npy.ones_like(V)

    HSV = npy.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    plt.figure()
    plt.imshow(npy.fliplr(npy.rot90(npy.flipud(RGB))),interpolation='none')
    plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+obj_file_name+'_hsv.png')

    plt.show()

if __name__ == '__main__':
    scan_num,sign = sys.argv[1:]
    save_hue(scan_num,sign)
