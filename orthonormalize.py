import numpy as npy
import matplotlib.pyplot as plt
import sys
import gramm

def dist(n):
    a = npy.arange(n)
    a = npy.where(a<npy.float(n)/2.,a,npy.abs(a-npy.float(n)))**2
    array=npy.zeros((n,n))
    for i in range(npy.int(n)/2+1):
        y=npy.sqrt(a+i**2)
        array[:,i]=y
        if i!=0:
            array[:,n-i]=y
    del(a)
    del(y)
    return npy.fft.fftshift(array)

def gram_schmidt(array1,array2):
    dot_product_1 = npy.vdot(array1,array2)
    dot_product_2 = npy.vdot(array1,array1)
    return array1*dot_product_1/dot_product_2


def orthonormalize(scan_num,sign, prb_or_obj):
    array = npy.load('./recon_result/S'+scan_num+'/'+sign+'/recon_data/recon_'+scan_num+'_'+sign+'_'+prb_or_obj+'_ave.npy')
    nx,ny,nz = npy.shape(array)

    if nx != ny:
        dim_max = npy.max((nx,ny))
        tmp = array.copy()
        array = npy.zeros((dim_max,dim_max,nz)).astype(complex)
        array[dim_max/2-nx/2:dim_max/2+nx/2,dim_max/2-ny/2:dim_max/2+ny/2,:] = tmp[:,:,:]

    plt.figure()
    plt.subplot(221)
    plt.imshow(npy.abs(array[:,:,0]))
    plt.colorbar()
    if nz >= 2:
        plt.subplot(222)
        plt.imshow(npy.abs(array[:,:,1]))
        plt.colorbar()
        if nz >=3:
            plt.subplot(223)
            plt.imshow(npy.abs(array[:,:,2]))
            plt.colorbar()
            if nz >=4:
                plt.subplot(224)
                plt.imshow(npy.abs(array[:,:,3]))
                plt.colorbar()

    power_1 = npy.zeros(nz)
    power_1_sort = npy.zeros(nz)
    for i in range(nz):
        power_1[i] = npy.sum(npy.abs(array[:,:,i])**2)
        power_1_sort[i] = power_1[i]

    power_1_sort.sort()

    array_sort = array.copy()
    for i in range(nz):
        index = npy.where(power_1 == power_1_sort[nz-1-i])
        array_sort[:,:,i] = array[:,:,index[0][0]]

    array_new = array_sort.copy()

    '''
    plt.figure()
    plt.subplot(221)
    plt.imshow(npy.abs(array_new[:,:,0]))
    plt.colorbar()
    if nz >= 2:
        plt.subplot(222)
        plt.imshow(npy.abs(array_new[:,:,1]))
        plt.colorbar()
        if nz >=3:
            plt.subplot(223)
            plt.imshow(npy.abs(array_new[:,:,2]))
            plt.colorbar()
            if nz >=4:
                plt.subplot(224)
                plt.imshow(npy.abs(array_new[:,:,3]))
                plt.colorbar()
    '''
    
    power_2 = npy.zeros(nz)
    for i in range(nz):
        power_1[i] = npy.sum(npy.abs(array_new[:,:,i])**2)
        for j in range(i):
            array_new[:,:,i] = array_new[:,:,i] - gram_schmidt(array_new[:,:,j],array_new[:,:,i])
        power_2[i] = npy.sum(npy.abs(array_new[:,:,i])**2)

    power_2_sort = power_2.copy()
    power_2_sort.sort()
    array_new_sort = array_new.copy()
    for i in range(nz):
        index = npy.where(power_2 == power_2_sort[nz-1-i])
        array_new_sort[:,:,i] = array_new[:,:,index[0][0]]

    plt.savefig('./recon_result/S'+scan_num+'/'+sign+'/recon_pic/'+scan_num+'_'+prb_or_obj+'_mode_ini.png')

    plt.figure()
    plt.subplot(221)
    plt.imshow(npy.abs(array_new_sort[:,:,0]))
    plt.colorbar()
    if nz >= 2:
        plt.subplot(222)
        plt.imshow(npy.abs(array_new_sort[:,:,1]))
        plt.colorbar()
        if nz >=3:
            plt.subplot(223)
            plt.imshow(npy.abs(array_new_sort[:,:,2]))
            plt.colorbar()
            if nz >=4:
                plt.subplot(224)
                plt.imshow(npy.abs(array_new_sort[:,:,3]))
                plt.colorbar()
    
    plt.savefig('./recon_result/S'+scan_num+'/'+sign+'/recon_pic/'+scan_num+'_'+prb_or_obj+'_mode_orth.png')

    plt.figure()
    plt.plot(power_1/npy.sum(power_1))
    plt.plot(power_1/npy.sum(power_1),'go')
    plt.ylim([0,1])
    plt.savefig('./recon_result/S'+scan_num+'/'+sign+'/recon_pic/'+scan_num+'_'+prb_or_obj+'_ini_mode_power.png')
    plt.figure()
    plt.plot(power_2_sort[::-1]/npy.sum(power_2_sort[::-1]))
    plt.plot(power_2_sort[::-1]/npy.sum(power_2_sort[::-1]),'go')
    plt.ylim([0,1])
    plt.savefig('./recon_result/S'+scan_num+'/'+sign+'/recon_pic/'+scan_num+'_'+prb_or_obj+'_mode_power.png')

    npy.save('./recon_result/S'+scan_num+'/'+sign+'/recon_data/recon_'+scan_num+'_'+sign+'_'+prb_or_obj+'_mode_orth_ave.npy',array_new_sort)

    plt.show()

if __name__ == '__main__':
    scan_num,sign, prb_or_obj = sys.argv[1:]
    orthonormalize(scan_num,sign, prb_or_obj)
