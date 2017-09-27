import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import sys

def dist(n):
    a = np.arange(n)
    a = np.where(a<np.float(n)/2.,a,np.abs(a-np.float(n)))**2
    array=np.zeros((n,n))
    for i in range(np.int(n/2)+1):
        y=np.sqrt(a+i**2)
        array[:,i]=y
        if i!=0:
            array[:,n-i]=y
    return np.fft.fftshift(array)

def cal_zp(diameter_m,finest_zone_width_m,energy_kev,defocal_distance_m,pixel_size_m,array_size,disp_flag=False):

    wavelength_m = (12.4 / energy_kev) * 1.e-10
    focal_length_m = diameter_m * finest_zone_width_m / wavelength_m

    n = 1024
    dr_m = diameter_m / 2 / n

    z_defocal_m = focal_length_m + defocal_distance_m

    array_size_new = np.int(array_size * 1.6)
    line_m = (np.arange(array_size_new) - array_size_new/2) * pixel_size_m
    k = 2 * np.pi / wavelength_m

    line = np.zeros(array_size_new).astype(complex)

    for ii in range(array_size_new):
        s = 0.
        for jj in range(n):
            r = jj * dr_m
            s += np.exp(1j * 0.5 * k * (1/z_defocal_m - 1/focal_length_m) * r**2) * \
              ss.j0(k * line_m[ii] * r / z_defocal_m) * r * dr_m
        line[ii] = s * np.exp(1j * 0.5 * k * line_m[ii]**2 / z_defocal_m)

    if disp_flag:
        plt.close('all')
        plt.figure()
        plt.subplot(221)
        plt.plot(np.abs(line))
        plt.title('Diameter: '+np.str(diameter_m/1e-6)+'um')
        plt.subplot(222)
        plt.plot(np.angle(line))
        plt.title('Outmost zone: '+np.str(finest_zone_width_m/1e-9)+'nm')

    zp_array = np.zeros((array_size,array_size)).astype(complex)
    dummy = dist(array_size)
    n_max = np.int(np.max(dummy))
    for i in range(n_max):
        tmp = line[np.int(array_size_new/2)+i]
        r_in = i
        r_out = i + 1
        index = np.where((dummy >= r_in) & (dummy < r_out))
        zp_array[index] = tmp

    if disp_flag:
        plt.subplot(223)
        plt.imshow(np.abs(zp_array))
        plt.title('Energy: '+np.str(energy_kev)+'keV')
        plt.subplot(224)
        plt.imshow(np.angle(zp_array))
        plt.title('Pixel size: '+np.str(pixel_size_m/1e-9)+'nm')
        plt.show()

    zp_array /= np.max(np.abs(zp_array))
    return zp_array

if __name__ == '__main__':
    diameter_m = np.float(sys.argv[1])
    finest_zone_width_m = np.float(sys.argv[2])
    energy_kev = np.float(sys.argv[3])
    defocal_distance_m = np.float(sys.argv[4])
    pixel_size_m = np.float(sys.argv[5])
    array_size = np.int(sys.argv[6])
    disp_flag = np.int(sys.argv[7])
    cal_zp(diameter_m,finest_zone_width_m,energy_kev,defocal_distance_m,pixel_size_m,array_size,disp_flag)
