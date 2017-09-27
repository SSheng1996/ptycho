import numpy as npy
import align_class as ac
import matplotlib.pyplot as plt
import sys
import os
import math
from matplotlib.colors import hsv_to_rgb

def rm_phase_ramp(scan_num,sign,flag=None,prb_only_flag=True):

    if flag == 'mode':
        try_num = sign
        dir_name = './recon_result/S'+scan_num+'/'+try_num+'/recon_data/'
        file_name = 'recon_'+scan_num+'_'+try_num
        prb_file_name = 'recon_'+scan_num+'_'+try_num+'_probe_mode_orth_ave'
        obj_file_name = 'recon_'+scan_num+'_'+try_num+'_object_mode_orth_ave'

        data_tmp = npy.load(dir_name+'/'+prb_file_name+'.npy')
        nx,ny,nz=npy.shape(data_tmp)

        if npy.mod(nx, 2) == 1:
            nx = nx - 1
        if npy.mod(ny, 2) == 1:
            ny = ny - 1

        for i in range(nz):
            prb_data = data_tmp[0:nx,0:ny,i]

            amp = npy.abs(prb_data)
            pha = npy.angle(prb_data)

            if prb_only_flag:
                prb_array = prb_data
            else:
                prb_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)
            npy.save(dir_name+'/'+prb_file_name+'_rp_mode_'+npy.str(i),prb_array)

            H_mean = npy.mean(npy.angle(prb_array))
            H = ((npy.angle(prb_array) - H_mean + math.pi) * 180. / math.pi)/360.

            V = npy.abs(prb_array) / npy.max(npy.abs(prb_array))
            S = npy.ones_like(V)

            HSV = npy.dstack((H,S,V))
            RGB = hsv_to_rgb(HSV)
            plt.figure()
            plt.imshow((npy.rot90(RGB)),interpolation='none')
            plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+prb_file_name+'_rp_mode_'+npy.str(i)+'.png')

            if i == 0:
                plt.figure()
                plt.subplot(221)
                plt.imshow(npy.flipud(npy.transpose(npy.abs(prb_array))),interpolation='none')
                plt.colorbar()
                plt.subplot(222)
                plt.imshow(npy.flipud(npy.transpose(npy.angle(prb_array))),interpolation='none')
                plt.colorbar()
                npy.save(dir_name+'/'+prb_file_name+'_rp',prb_array)

        data_tmp = npy.load(dir_name+'/'+obj_file_name+'.npy')
        nx,ny,nz=npy.shape(data_tmp)

        if npy.mod(nx, 2) == 1:
            nx = nx - 1
        if npy.mod(ny, 2) == 1:
            ny = ny - 1

        for i in range(nz):
            obj_data = data_tmp[0:nx,0:ny,i]

            amp = npy.abs(obj_data)
            pha = npy.angle(obj_data)
            obj_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)
            npy.save(dir_name+'/'+obj_file_name+'_rp_mode_'+npy.str(i),obj_array)

            H_mean = npy.mean(npy.angle(obj_array))
            H = ((npy.angle(obj_array) - H_mean + math.pi) * 180. / math.pi)/360.

            V = npy.abs(obj_array) / npy.max(npy.abs(obj_array))
            S = npy.ones_like(V)

            HSV = npy.dstack((H,S,V))
            RGB = hsv_to_rgb(HSV)
            plt.figure()
            plt.imshow((npy.rot90(RGB)),interpolation='none')
            plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+obj_file_name+'_rp_mode_'+npy.str(i)+'.png')

            if i == 0:
                plt.figure()
                plt.subplot(223)
                plt.imshow(npy.flipud(npy.transpose(npy.abs(obj_array))),interpolation='none')
                plt.colorbar()
                plt.subplot(224)
                plt.imshow(npy.flipud(npy.transpose(npy.angle(obj_array))),interpolation='none')
                plt.colorbar()
                npy.save(dir_name+'/'+obj_file_name+'_rp',obj_array)
                plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+file_name+'_rp.png')

        plt.show()

    elif flag == 'ms':
        try_num = sign
        dir_name = './recon_result/S'+scan_num+'/'+try_num+'/recon_data/'
        file_name = 'recon_'+scan_num+'_'+try_num
        prb_file_name = 'recon_'+scan_num+'_'+try_num+'_probe_ave'
        obj_file_name = 'recon_'+scan_num+'_'+try_num+'_object_ave'

        data_tmp = npy.load(dir_name+'/'+obj_file_name+'.npy')
        nx,ny,nz=npy.shape(data_tmp)

        if npy.mod(nx, 2) == 1:
            nx = nx - 1
        if npy.mod(ny, 2) == 1:
            ny = ny - 1

        for i in range(nz):
            obj_data = data_tmp[0:nx,0:ny,i]

            amp = npy.abs(obj_data)
            pha = npy.angle(obj_data)
            obj_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)
            npy.save(dir_name+'/'+obj_file_name+'_rp_ms_'+npy.str(i),obj_array)

            H_mean = npy.mean(npy.angle(obj_array))
            H = ((npy.angle(obj_array) - H_mean + math.pi) * 180. / math.pi)/360.

            V = npy.abs(obj_array) / npy.max(npy.abs(obj_array))
            S = npy.ones_like(V)

            HSV = npy.dstack((H,S,V))
            RGB = hsv_to_rgb(HSV)
            plt.figure()
            plt.imshow((npy.rot90(RGB)),interpolation='none')
            plt.title('object '+npy.str(i))
            plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+obj_file_name+'_rp_ms_'+npy.str(i)+'.png')

            if i == 0:
                plt.figure()
                plt.subplot(223)
                plt.imshow(npy.flipud(npy.transpose(npy.abs(obj_array))),interpolation='none')
                plt.colorbar()
                plt.title('object 0 amplitude')
                plt.subplot(224)
                plt.imshow(npy.flipud(npy.transpose(npy.angle(obj_array))),interpolation='none')
                plt.colorbar()
                plt.title('object 0 phase')
                npy.save(dir_name+'/'+obj_file_name+'_rp',obj_array)
                plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+file_name+'_rp.png')

        data_tmp = npy.load(dir_name+'/'+prb_file_name+'.npy')
        #nx,ny,nz=npy.shape(data_tmp)

        #if npy.mod(nx, 2) == 1:
        #    nx = nx - 1
        #if npy.mod(ny, 2) == 1:
        #    ny = ny - 1

        for i in range(nz):
            #prb_data = data_tmp[0:nx,0:ny,i]
            prb_data = data_tmp[0][i]
            nx,ny = npy.shape(prb_data)

            amp = npy.abs(prb_data)
            pha = npy.angle(prb_data)

        if prb_only_flag:
            prb_array = prb_data
        else:
            prb_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)
            npy.save(dir_name+prb_file_name+'_rp_ms_'+npy.str(i),prb_array)

            H_mean = npy.mean(npy.angle(prb_array))
            H = ((npy.angle(prb_array) - H_mean + math.pi) * 180. / math.pi)/360.

            V = npy.abs(prb_array) / npy.max(npy.abs(prb_array))
            S = npy.ones_like(V)

            HSV = npy.dstack((H,S,V))
            RGB = hsv_to_rgb(HSV)
            plt.figure()
            plt.title('probe '+npy.str(i))
            plt.imshow((npy.rot90(RGB)),interpolation='none')
            plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+prb_file_name+'_rp_ms_'+npy.str(i)+'.png')

            if i == 0:
                plt.figure()
                plt.subplot(221)
                plt.imshow(npy.flipud(npy.transpose(npy.abs(prb_array))),interpolation='none')
                plt.title('probe 0 amplitude')
                plt.colorbar()
                plt.subplot(222)
                plt.imshow(npy.flipud(npy.transpose(npy.angle(prb_array))),interpolation='none')
                plt.colorbar()
                plt.title('probe 0 phase')
                npy.save(dir_name+'/'+prb_file_name+'_rp',prb_array)


        plt.show()

    else:
        try_num = sign

        dir_name = './recon_result/S'+scan_num+'/'+try_num+'/recon_data/'
        file_name = 'recon_'+scan_num+'_'+try_num
        prb_file_name = 'recon_'+scan_num+'_'+try_num+'_probe_ave'
        obj_file_name = 'recon_'+scan_num+'_'+try_num+'_object_ave'

        data_tmp = npy.load(dir_name+'/'+prb_file_name+'.npy')
        nx,ny=npy.shape(data_tmp)

        if npy.mod(nx, 2) == 1:
            nx = nx - 1
        if npy.mod(ny, 2) == 1:
            ny = ny - 1

        prb_data = data_tmp[0:nx,0:ny]

        amp = npy.abs(prb_data)
        pha = npy.angle(prb_data)

        if prb_only_flag:
            prb_array = prb_data
        else:
            prb_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)

        H_mean = npy.mean(npy.angle(prb_array))
        H = ((npy.angle(prb_array) - H_mean + math.pi) * 180. / math.pi)/360.

        V = npy.abs(prb_array) / npy.max(npy.abs(prb_array))
        S = npy.ones_like(V)

        HSV = npy.dstack((H,S,V))
        RGB = hsv_to_rgb(HSV)
        plt.figure()
        plt.imshow((npy.rot90(RGB)),interpolation='none')
        plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+prb_file_name+'_rp.png')

        data_tmp = npy.load(dir_name+'/'+obj_file_name+'.npy')
        nx,ny=npy.shape(data_tmp)

        if npy.mod(nx, 2) == 1:
            nx = nx - 1
        if npy.mod(ny, 2) == 1:
            ny = ny - 1

        obj_data = data_tmp[0:nx,0:ny]

        amp = npy.abs(obj_data)
        pha = npy.angle(obj_data)

        obj_array = ac.remove_phase_ramp(amp * npy.exp(1j*pha),0,0.1,1)

        H_mean = npy.mean(npy.angle(obj_array))
        H = ((npy.angle(obj_array) - H_mean + math.pi) * 180. / math.pi)/360.

        V = npy.abs(obj_array) / npy.max(npy.abs(obj_array))
        S = npy.ones_like(V)

        HSV = npy.dstack((H,S,V))
        RGB = hsv_to_rgb(HSV)
        plt.figure()
        plt.imshow((npy.rot90(RGB)),interpolation='none')
        plt.savefig('./recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'+obj_file_name+'_rp.png')

        plt.figure()
        plt.subplot(221)
        plt.imshow(npy.flipud(npy.transpose(npy.abs(prb_array))),interpolation='none')
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(npy.flipud(npy.transpose(npy.angle(prb_array))),interpolation='none')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(npy.flipud(npy.transpose(npy.abs(obj_array))),interpolation='none')
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(npy.flipud(npy.transpose(npy.angle(obj_array))),interpolation='none')
        plt.colorbar()

        save_pic_dir = './recon_result/S'+scan_num+'/'+try_num+'/recon_pic/'
        if not os.path.exists(save_pic_dir):
            os.makedirs(save_pic_dir)
        plt.savefig(save_pic_dir+'/'+file_name+'_rp.png')

        npy.save(dir_name+'/'+prb_file_name+'_rp',prb_array)
        npy.save(dir_name+'/'+obj_file_name+'_rp',obj_array)

        plt.show()

if __name__ == '__main__':
    scan_num,sign,flag = sys.argv[1:4]
    rm_phase_ramp(scan_num,sign,flag)
