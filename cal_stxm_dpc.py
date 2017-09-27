import numpy as np
import align_class as ac
import time

def pad_gradient(g):
    nx,ny = np.shape(g)
    gg = np.zeros((2*nx,2*ny))
    gg[:nx,:ny] = g
    gg[nx:,:ny] = np.flipud(g)
    gg[:nx,ny:] = np.fliplr(g)
    gg[nx:,ny:] = np.flipud(np.fliplr(g))
    return gg


def phase_uw(gx,gy,dx,dy):
    row,column = np.shape(gx)
    '''
    nx,ny = np.shape(gx)
    #dx = 1.
    #dy = 1.
    gx_tmp = gx.copy()
    gy_tmp = gy.copy()

    gx = np.zeros((nx+40,ny+40))
    gy = np.zeros((nx+40,ny+40))

    gx[20:-20,20:-20] = gx_tmp
    gy[20:-20,20:-20] = gy_tmp

    row,column = np.shape(gx)
    '''
    w = 1. # Weighting parameter
    tx = np.fft.fftshift(np.fft.fft2(gx)) / np.sqrt(np.size(gx))
    ty = np.fft.fftshift(np.fft.fft2(gy))/ np.sqrt(np.size(gy))
    '''
    tx_tmp = tx.copy()
    ty_tmp = ty.copy()

    tx = np.zeros((row+40,column+40)).astype(complex)
    ty = np.zeros((row+40,column+40)).astype(complex)
    tx[20:-20,20:-20] = tx_tmp
    ty[20:-20,20:-20] = ty_tmp

    row = row + 40
    column = column + 40
    '''
    c = np.arange(row*column, dtype=complex).reshape(row, column)
    for i in range(row):
        for j in range(column):
            kappax = 2*np.pi * (i-np.floor(row/2.0)) / (row*dx)
            kappay = 2*np.pi * (j-np.floor(column/2.0)) / (column*dy)
            #kappax = (i-np.floor(row/2.0)) / (row*dx)
            #kappay = (j-np.floor(column/2.0)) / (column*dy)
            if kappax==0 and kappay==0:
                c[i, j] = 0
            else:
                cTemp = 1j * (kappax*tx[i][j]+w*kappay*ty[i][j]) / (kappax**2 + w*kappay**2 + 1.e-12)
                c[i, j] = cTemp
    c = np.fft.ifftshift(c)
    phi = np.fft.ifft2(c) * np.sqrt(np.size(gx))

    #phi = phi[20:-20,20:-20]
    #print('phase error: ', np.sum(np.abs(phi.imag)))

    return phi.real

def cal_stxm_dpc(data, det_pix_um, z_m, lambda_nm, x_step_m, y_step_m, c, r,crop_size=64,x_flip=1,y_flip=1):

    time_start = time.time()
    nz,nx,ny = np.shape(data)
    stxm = np.zeros(nz)

    x_shift = np.zeros(nz)
    y_shift = np.zeros(nz)

    det_pix_m = det_pix_um * 1.e-6
    lambda_m = lambda_nm * 1.e-9

    for i in range(nz):
        if np.mod(i,500) == 0:
            print('calculating frame ',i)
        tmp = np.fft.fftshift(data[i,:,:])
        tmp = tmp[nx/2-crop_size/2:nx/2+crop_size/2,ny/2-crop_size/2:ny/2+crop_size/2]
        stxm[i] = np.sqrt(np.sum(tmp**2))

        zero_int_sign = 0
        if i==0:
            ref = np.fft.fftn(tmp)
            s = ref
        else:
            if stxm[i] == 0:
                s = ref
                zero_int_sign = 1
                stxm[i] = stxm[i-1]
            else:
                s = np.fft.fftn(tmp)
        e, p, x_shift[i], y_shift[i], array_shift = ac.dftregistration(ref,s,usfac=1000)
        if zero_int_sign:
            x_shift[i] = x_shift[i-1]
            y_shift[i] = y_shift[i-1]

    stxm /= np.max(stxm)

    stxm.resize(r,c)
    x_shift.resize(r,c)
    y_shift.resize(r,c)

    stxm = np.fliplr(stxm.T)
    x_shift = np.fliplr(x_shift.T)
    y_shift = np.fliplr(y_shift.T)

    k = 2*np.pi/lambda_m

    gx = x_shift * det_pix_m * k / z_m
    gy = y_shift * det_pix_m * k / z_m

    ggx = pad_gradient(gx)
    ggy = pad_gradient(gy)

    phase_pad = phase_uw(ggx,ggy, x_step_m, y_step_m)

    if x_flip == 1:
        if y_flip == 1:
            phase = phase_pad[:c,:r]
        else:
            phase = np.fliplr(phase_pad[:c,r:])
    else:
        if y_flip == 1:
            phase = np.flipud(phase_pad[c:,:r])
        else:
            phase = np.flipud(np.fliplr(phase_pad[c:,r:]))

    time_end = time.time()
    print('dpc cal takes ',time_end-time-start,'second')
    return stxm*np.exp(1j*phase),phase_pad
