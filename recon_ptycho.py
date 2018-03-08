from ptycho_trans_ml import ptycho_trans
import numpy as np
import sys
from scipy import interpolate
import math
import scipy.misc as sm
import matplotlib.pyplot as plt
import tifffile as tf
import os
import rot
import prop_class as pc
import simulate_zp as szp
import h5py
from mpi4py import MPI

def shift_sum(array_in,x_range):
    x_shift = np.arange(x_range+1) - x_range/2
    array_out = np.zeros_like(array_in)
    for i in range(x_range+1):
        array_out += np.roll(array_in,np.int(x_shift[i]),0)
    return array_out


def congrid_fft(array_in, shape):
    x_in, y_in = np.shape(array_in)
    array_in_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(array_in)))/np.sqrt(1.*x_in*y_in)
    array_in_fft_n = np.zeros((shape[0],shape[1])).astype(complex)
    if x_in < shape[0]:
        array_in_fft_n[shape[0]//2-x_in//2:shape[0]//2+x_in//2,shape[1]//2-y_in//2:shape[1]//2+y_in//2] \
                = array_in_fft
    else:
        array_in_fft_n \
                = array_in_fft[x_in//2-shape[0]//2:shape[0]//2+x_in//2,y_in//2-shape[1]//2:shape[1]//2+y_in//2]

    array_out =  np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(array_in_fft_n)))*np.sqrt(1.*shape[0]*shape[1])
    return array_out

def recon(scan_num,sign,n_iterations,p_flag, gpu_flag):

    print('enter recon')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    lrank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    lsize=int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    print ("my rank ,local rank is local_size  Total rank  : ", rank ,lrank, lsize, size )
    

    mode_flag = 1
    mesh_flag = 1
    distance = 0
    x_range=np.int64(0)
    y_range=np.int64(0)
    dr_x=np.float64(0.0)
    dr_y=np.float64(0.0)
    z_m=np.float64(0.0)
    lambda_nm=np.float64(0.0)
    nz=0
    list=None
    
    
    
    if rank == 0 :
       f = h5py.File('scan_'+scan_num+'.h5','r')
       diffamp = np.array(f['diffamp'])         # scan data
       points = np.array(f['points'])
       x_range = np.array(f['x_range'])
       y_range = np.array(f['y_range'])
       dr_x = np.array(f['dr_x'])
       dr_y = np.array(f['dr_y'])
       z_m = np.array(f['z_m'])
       lambda_nm = np.array(f['lambda_nm'])
       f.close()

       nz, nx , ny =np.shape(diffamp)
       print ("shape of diffamp: ", np.shape(diffamp) )
       
       diff_all=np.array_split(diffamp, size)
       list=[] 
       for i in range(size):
            list.append(np.shape(diff_all[i]))

    
    list=comm.bcast(list, root=0 )
    nz=comm.bcast(nz,root=0)
    x_range=comm.bcast(x_range,root=0)
    y_range=comm.bcast(y_range,root=0)
    dr_x=comm.bcast(dr_x,root=0)
    dr_y=comm.bcast(dr_y,root=0)
    z_m=comm.bcast(z_m,root=0)
    lambda_nm=comm.bcast(lambda_nm,root=0)


    if rank == 0 :

       diff_l=diff_all[0] 
       for i in range(1, size) :
            comm.Send(diff_all[i], dest=i , tag=i+70 ) 

    else : 
       diff_l = np.empty(list[rank], dtype=np.float64)
       comm.Recv(diff_l, source = 0, tag = rank+70 ) 
    nz_l, nx , ny =np.shape(diff_l)
    print ("shape of diff_l: ",rank, np.shape(diff_l) )
    print (rank , x_range, y_range , dr_x, dr_y , z_m, lambda_nm,nz_l )        



 
    kernal_n = 32
    if rank == 0:
        recon = ptycho_trans(diffamp)  
    else:
        recon = ptycho_trans(diff_l)
    recon.comm = comm
    recon.size = size
    recon.rank = rank
    recon.lrank = lrank
    recon.lsize = lsize
    recon.ptdist = list
    recon.diff_l = diff_l

    recon.file_num = nz            
    recon.nx_prb = nx              
    recon.ny_prb = ny              
    recon.num_points = nz 
    recon.num_points_l = nz_l

    recon.start_ave = 0.8
    recon.scan_num = scan_num  #scan number
    recon.online_flag = False
    #recon.tif_threshold = threshold
    recon.sign = sign      #saving file name
    recon.x_range_um = x_range        #scan range in x direction (um)
    recon.y_range_um = y_range        #scan range in y direction (um)
    if mesh_flag:
        recon.mesh_flag = True      #True to use mesh scan pattern, False to use spiral scan
        recon.fermat_flag = False
    else:
        recon.mesh_flag = False
        recon.fermat_flag = True
    recon.x_dr_um = dr_x           #scan step size in x direction (um)
    recon.y_dr_um = dr_y           #scan step size in y direction (um)
    recon.dr_um = dr_x//1            #radius increment size (um)
    #recon.dr_um = 0.75           #radius increment size (um)
    recon.nth = 5.                #number of points in first ring
    recon.x_roi = nx              #data array size in x
    recon.y_roi = ny              #data array size in y
    recon.lambda_nm = lambda_nm      #x-ray wavelength (nm)
    recon.z_m = z_m                #detector-to-sample distance (m)
    recon.ccd_pixel_um = 55.      #detector pixel size (um)

    # scan direction and geometry correction handling
    if rank ==0 :
        recon.points = -1*points
        recon.points[0,:] *=-1 * np.abs(np.cos(15.*np.pi/180.))

    # recon pixel size
    pixel_size_m = recon.lambda_nm * recon.z_m * 1e-3 / (recon.x_roi * recon.ccd_pixel_um)

    recon.alg_flag = 'DM'  # choose from 'DM', 'ML', 'ER'
    recon.ml_mode = 'Poisson' # mode for ML
    recon.alg2_flag = 'DM' # choose from 'DM', 'ML', 'ER'
    recon.alg_percentage = 0.8

    # param for Bragg mode
    recon.bragg_flag = False
    recon.bragg_theta = 69.41
    recon.bragg_gamma = 33.4
    recon.bragg_delta = 15.458

    recon.amp_max = 1.            #up limit of allowed object amplitude range
    recon.amp_min = 0.85       #low limit of allowed object amplitude range
    recon.pha_max = 0.01       #up limit of allowed object phase range
    recon.pha_min = -0.6


    #parameters for partial coherence calculation
    pc_flag = False
    recon.pc_flag = pc_flag
    recon.update_coh_flag = pc_flag
    recon.init_coh_flag = pc_flag
    recon.pc_kernel_n = kernal_n           #kernal size
    #print(recon.pc_kernel_n)
    recon.pc_sigma = 2          #initial guess of kernal sigma
    recon.pc_alg = 'lucy'

    #reconstruction feedback parameters
    recon.alpha = 1.e-8
    recon.beta = 0.9

    recon.mode_flag = mode_flag
    recon.save_tmp_pic_flag = False
    recon.prb_mode_num = 5
    recon.obj_mode_num = 1

    recon.dm_version = 1
    recon.position_correction_flag = False

    recon.multislice_flag = False
    if recon.multislice_flag:
        recon.slice_num = 2
        recon.slice_spacing_m = 5.e-6

    recon.n_iterations = n_iterations   #number of iterations
    recon.start_update_probe = 0              #iteration number for probe updating
    recon.start_update_object = 0

    recon.init_obj_flag = True               #True to start with a random guess. False to load a pre-existing array
    recon.init_obj_dpc_flag = False
    recon.dpc_x_step_m = recon.x_dr_um * 1.e-6
    recon.dpc_y_step_m = recon.y_dr_um * 1.e-6

    #recon.prb_center_flag = True
    recon.init_prb_flag = False             #True to start with a random guess. False to load a pre-existing array
    recon.mask_prb_flag = False
    if rank == 0 :
        if p_flag:
            prb = np.load('./recon_result/S'+scan_num+'/t1/recon_data/recon_'+scan_num+'_t1_probe_ave_rp.npy')
        else:
            #s = diffamp[0,:,:]
            #tmp = np.fft.fftshift(np.fft.fftn(s)) / np.sqrt(np.size(s))
            tmp = np.load('recon_probe.npy')
            prb = tmp

        if prb.shape != (nx, ny):
            print('Resizing loaded probe from %s to %s' % (prb.shape, (nx, ny)))
            prb = congrid_fft(prb,(nx,ny))

        if not mode_flag:
            recon.prb = prb
        else:
            recon.start_update_probe = 0
            recon.prb_mode =  np.zeros((recon.prb_mode_num,recon.nx_prb, recon.ny_prb)).astype(complex)
            recon.prb_mode[0,:,:] = prb
            recon.prb_mode[1,:,:] = shift_sum(prb,20) / 40.
            recon.prb_mode[2,:,:] = shift_sum(prb,16) / 32.
            recon.prb_mode[3,:,:] = shift_sum(prb,10) / 20.
            recon.prb_mode[4,:,:] = shift_sum(prb,6) / 12.

    recon.sf_flag = False

    recon.recon_code = __file__        #Copy the code

    recon.position_correction_start = 50
    recon.position_correction_step = 10

    recon.processes = 0
    recon.gpu_flag = gpu_flag
    recon.recon_ptycho()
    #if rank == 0 :
    #    recon.recon_ptycho()
    #if rank == 0 : 
        #recon.save_recon()
        #recon.display_recon()



if __name__ == '__main__':
    scan_num,sign = sys.argv[1:3]
    n_iterations =  np.int(sys.argv[3])
    p_flag = np.int(sys.argv[4])
    gpuflag = np.int(sys.argv[5])
    recon(scan_num,sign,n_iterations,p_flag,gpuflag)
