import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni

def rot(array,angle):
    nx, ny = np.shape(array)

    data = sni.rotate(array,angle)

    nxx, nyy = np.shape(data)
    return data[(nxx-nx)/2:(nxx-nx)/2+nx,(nyy-ny)/2:(nyy-ny)/2+ny]
