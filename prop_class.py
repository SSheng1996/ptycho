import numpy as npy
import scipy.fftpack as sf
import math
import matplotlib.pyplot as plt

def qpf(prop, x_pixel_size, y_pixel_size, wavelength, z):
    ndim = npy.ndim(prop)
    if ndim == 1:
        nx = npy.size(prop)
        ny =1
    if ndim == 2:
        nx,ny = npy.shape(prop)

    half_nx = nx / 2
    half_ny = ny / 2

    xi_max = 1./(2.*x_pixel_size)
    eta_max = 1./(2.*y_pixel_size)
    xi_arr = npy.arange(nx) - half_nx
    xi_arr = xi_arr / (2. * x_pixel_size * half_nx)
    xi_arr_l = xi_arr * wavelength
    #for i in range(nx):
    #    xi_arr[i,:] = tmp[i]

    eta_arr = npy.arange(ny) - half_ny
    #if ny <= 2:
    #    eta_arr = npy.arange(ny)
    #else:
    eta_arr =  eta_arr / (2. * y_pixel_size * half_ny)
    eta_arr_l = eta_arr * wavelength
    #    for i in range(ny):
    #        eta_arr[:,i] = tmp[i]


    trans_x2 = xi_arr_l * xi_arr * z
    trans_y2 = eta_arr_l * eta_arr * z
    trans_x4 = trans_x2 * xi_arr_l * xi_arr_l / 4.
    trans_y4 = trans_y2 * eta_arr_l * eta_arr_l / 4.
    trans_x6 = trans_x4 * xi_arr_l * xi_arr_l /2.
    trans_y6 = trans_y4 * eta_arr_l * eta_arr_l / 2.
    #trans_x2y2 = trans_x2 * eta_arr_l * eta_arr_l / 4.
    #trans_x4y2 = trans_x4 * eta_arr_l * eta_arr_l * 3. / 8.
    #trans_x2y4 = trans_y4 * xi_arr_l * xi_arr_l * 3. / 8.

    phase_tolerance = 1./16.
    phase = npy.ones((nx,ny))
    phase_non_fresnel = trans_x4[nx-1] + trans_y4[ny-1] + \
        trans_x2[nx-1] * eta_arr_l[ny-1] * eta_arr_l[ny-1]
    if phase_non_fresnel >= phase_tolerance:
        for i_ny in range(ny):
            trans_x2y2 = trans_x2 * eta_arr_l[i_ny] * eta_arr_l[i_ny] * 2.
            trans_x4y2 = trans_x4 * eta_arr_l[i_ny] * eta_arr_l[i_ny] * 3. / 2.
            trans_x2y4 = trans_y4[i_ny] * xi_arr_l * xi_arr_l * 3. / 2.
            #phase[:,i_ny] = -1. * npy.pi *(-trans_x2 - trans_y2[i_ny] - \
            #                               trans_x4 - trans_y4[i_ny] - \
            #                               trans_x2y2 - trans_x6 - \
            #                               trans_y6[i_ny] - trans_x4y2 - \
            #                               trans_x2y4)
            phase[:,i_ny] = npy.pi *(-trans_x2 - trans_y2[i_ny] - \
                    trans_x4 - trans_y4[i_ny] - \
                    trans_x2y2 - trans_x6 - \
                    trans_y6[i_ny] - trans_x4y2 - \
                    trans_x2y4)
    else:
        for i_ny in range(ny):
            #phase[:,i_ny] = -1. * npy.pi * (-trans_x2 - trans_y2[i_ny])
            phase[:,i_ny] = npy.pi * (-trans_x2 - trans_y2[i_ny])


    tmp = npy.exp(1j*phase)

    if ndim == 1:
        prop = npy.zeros(nx).astype(complex)
        prop = tmp[:,0]
    else:
        prop = tmp

    return prop


def propagate(array, direction, x_pixel_size_m, y_pixel_size_m, wavelength_m, z_m):
    ndim = npy.ndim(array)
    if ndim == 1:
        nx = npy.size(array)
        prop = npy.zeros(nx).astype(complex)
    if ndim == 2:
        nx,ny = npy.shape(array)
        prop = npy.zeros((nx,ny)).astype(complex)

    prop = qpf(prop, x_pixel_size_m, y_pixel_size_m, wavelength_m, z_m)
    #npy.save('prop',prop)
    prop = sf.fftshift(prop)

    if direction == -1:
        prop = prop.conjugate()

    array = sf.ifftn(sf.fftshift(array)) * prop
    array = sf.fftshift(sf.fftn(array))

    return array

def multiprop(delta_beta_volume, voxel_size_m, wavelength_m,complex_flag):
    if(complex_flag == 0):
        nx,ny,nz,tmp = npy.shape(delta_beta_volume)
    if(complex_flag == 1):
        nx,ny,nz= npy.shape(delta_beta_volume)

    prop = npy.zeros((nx,ny)).astype(complex)
    prop = qpf(prop, voxel_size_m, voxel_size_m, wavelength_m, voxel_size_m)
    wavefield = npy.ones((nx,ny)) + 1j*npy.ones((nx,ny))

    kt = 2. * math.pi * voxel_size_m / wavelength_m

    for iz in range(nz):
        if(complex_flag == 0):
            delta_slice = delta_beta_volume[:,:,iz,0]
            beta_slice = delta_beta_volume[:,:,iz,0]
        if(complex_flag == 1):
            delta_slice = delta_beta_volume[iz,:,:].real
            beta_slice = delta_beta_volume[iz,:,:].imag
        wavefield = wavefield * npy.exp(-1.*kt*beta_slice+1j*kt*delta_slice)
        ft_wavefield = sf.ifftshift(sf.ifftn(sf.fftshift(wavefield))) * prop
        wavefield = sf.ifftshift(sf.fftn(sf.fftshift(ft_wavefield)))

    return wavefield
