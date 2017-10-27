import numpy as np
from scipy import rand
import time
import math
import matplotlib.pyplot as plt
import os
import scipy.misc as sm
from scipy import interpolate
#import pyfftw
import traceback
import multiprocessing as mp
import prop_class as pc
import copy
import tifffile as tf

import scipy.fftpack as sf
import shutil
import sys
import orthonormalize as orth
from save_hue_auto import save_hue
from rm_phase_ramp import rm_phase_ramp
from scipy.signal import convolve2d
from cal_stxm_dpc import cal_stxm_dpc
#from rm_phase_ramp_recon_all_modes import rm_phase_ramp_mode
#from rm_phase_ramp_recon_all import rm_phase_ramp


## For GPU 
import pycuda.autoinit
import pycuda.driver as cuda
from  pycuda.compiler import SourceModule
from pycuda import gpuarray, compiler, tools 
import skcuda.fft as cu_fft



if 'PROFILE' not in os.environ:
    def profile(fcn):
        return fcn

def parallel_function(fcn_name, *args, **kwargs):
    try:
        instance = ptycho_trans.instance
        fcn = getattr(instance, fcn_name)
        return fcn(*args, **kwargs)
    except KeyboardInterrupt:
        print('Cancelled')
    except Exception as ex:
        print('Parallel execution of `%s` failed: (%s) %s' % (fcn_name, ex.__class__.__name__, ex))
        traceback.print_exc()


class ptycho_trans(object):
    def __init__(self, diffamp):
        ptycho_trans.instance = self

        self.verbose = True

        # public attributes
        self.diff_array = diffamp  ##diffraction data
        self.nx_prb = None  ##number of pixels per side of array
        self.ny_prb = None  ##number of pixels per side of array
        self.num_points = None  ##number of scan points
        self.nx_obj = None  ## object x dimension
        self.ny_obj = None  ## object y dimension
        self.prb = None  ## probe array
        self.obj = None  ## object array
        self.prb_ave = None  ## averaged porbe
        self.obj_ave = None  ## averaged object
        self.prb_old = None  ## previous probe
        self.obj_old = None  ## previous object
        self.sign = None  ## saving file name
        self.scan_num = None ## scan number
        self.init_obj_flag = True  ## initialize random object flag
        self.init_obj_dpc_flag = True ## initialize object using stxm and dpc
        self.init_prb_flag = True  ## initialize random probe flag
        self.update_product_flag = False ## update product flag
        self.start_update_product = 10 ## start update product
        self.product = None  ## product array
        self.beta = 1. ##general feedback parameter
        self.alpha = 1.e-8  ##espresso threshold coefficient
        self.n_iterations = 10000 ##number of iterations
        self.start_update_probe = 2 ## iteration number start updating probe
        self.start_update_object = 0 ## iteration number start updating object
        self.end_update_probe = self.n_iterations  ## iteration number ends updating probe
        self.search_range = 10  ##search range for centering
        self.points = None  ## scan pattern
        self.point_info = None ## x,y start and end indecies
        self.sigma1 = 1.e-10  ## normalization weighting factor 1
        self.sigma2 = 5.e-5   ## normalization weighting factor
        self.amp_max = None  ## maximum object magnitude
        self.amp_min = None  ## minimum object magnitude
        self.pha_max = None  ## maximum object phase
        self.pha_min = None  ## minimum object phase
        self.ave_i = 0  ## average number
        self.error_obj = None ## chi square error for object
        self.error_prb = None ## chi square error for probe
        self.time_start = None  ## start time
        self.time_end = None  ## end time
        self.start_ave = 0.8  ## average starting iteration
        self.display_error_flag = True ## display reconstruction result flag
        self.sf_flag = True ## use FFTW or numpy fft
        self.x_direction_flag = False
        self.save_tmp_pic_flag = True
        self.x_dr_um = None
        self.y_dr_um = None
        self.mesh_flag = False
        self.dm_version = None
        self.recon_code = None
        self.cal_scan_pattern_flag = False
        self.prb_center_flag = False
        self.x_direction = 1
        self.y_direction = 1
        self.alg_percentage = 0.7
        self.obj_pad = 30

        self.dpc_x_step_m = 50.e-9
        self.dpc_y_step_m = 50.e-9
        self.dpc_crop_size = 64
        self.dpc_x_flip = 1
        self.dpc_y_flip = 1
        self.dpc_col = 100
        self.dpc_row = 100

        self.online_flag = False
        self.file_dir = None
        self.file_num = None
        self.x_c = None
        self.y_c = None
        self.tif_threshold = None

        # maximum likelihood paramter
        self.alg_flag = 'DM'     # option from DM, ER, ML
        self.ml_weight = 0.1     # weight factor for ML
        self.ml_mode = 'Poisson' # ML mode, option from Gaussian, Poisson
        self.alg2_flag = False      # run last 30% iterations using DM, ER or ML

        # experimental parameter
        self.x_range_um = None   # x scan range
        self.y_range_um = None   # y scan range
        self.dr_um = None        # radius increment
        self.nth = None          # number of points in the first ring
        self.x_roi = None        # x roi
        self.y_roi = None        # y roi
        self.lambda_nm = None    # wavelength
        self.z_m = None          # ccd distance
        self.ccd_pixel_um = None # ccd pixel size

        # partial coherence parameter
        self.pc_flag = True          # use partial coherence or not
        self.init_coh_flag = True
        self.init_pc_filter_flag = True
        self.coh = None          # deconvolution kernal
        self.diffint_center = None    # central diffraction data array
        self.pc_kernel_n = 32       # kernal size
        self.pc_sigma = 0.1      # kernal width
        self.lucy_it_num = 10          # number of iteration for kernal updating loop
        self.pc_step = 10    # how often to update coherence function
        self.pc_start = 0.1
        self.pc_end = 0.8
        self.update_coh_flag = True  # update coherence function or not
        self.pc_modulus_flag = False
        self.coh_percent = 0.5                # percentage of points used for coherence updating
        self.coh_old = None  ## previous object
        self.error_coh = None ## chi square error for coherence
        self.pc_alg = 'lucy'   # deconvolution algorithm, selection from 'lucy', 'wiener'.
        self.pc_wiener_factor = 0.1
        self.pc_filter = None

        # mode calculation parameter
        self.mode_flag = False
        self.prb_mode_num = 1 # number of probe modes
        self.obj_mode_num = 1 # number of probe modes
        self.prb_mode = None  # probe mode array
        self.obj_mode = None  # probe mode array
        self.prb_mode_ave = None
        self.obj_mode_ave = None
        self.prb_mode_old = None
        self.pobj_mode_old = None
        self.error_prb_mode = None
        self.error_obj_mode = None

        self.bragg_flag = False
        self.bragg_theta = None
        self.bragg_delta = None
        self.bragg_gamma = None

        # position correction parameter
        self.position_correction_flag = False
        self.position_correction_flag_ini = False
        self.position_correction_search_range = 1
        self.position_correction_start = 100
        self.position_correction_step = 10
        self.points_ini = None
        self.points_list = None

        # multislice parameter
        self.multislice_flag = False
        self.slice_num = 2
        self.slice_spacing_m = 10.e-6
        self.prb_ms = None
        self.obj_ms = None
        self.prb_ms_ave = None
        self.obj_ms_ave = None
        self.prb_ms_old = None
        self.obj_ms_old = None
        self.error_prb_ms = None
        self.error_obj_ms = None

        #GPU
        self.gpu_flag = False
	self.prb_d = None
        self.points_info_d = None
        self.obj_d = None
        self.product_d = None
        self.fft_tmp_d = None
        self.points_info_d = None
        self.prb_obj_d = None 
        self.diff_d = None
        self.amp_tmp_d = None
        self.power_d = None
        self.dev_buff_d = None
        self.dev_d = None
        self.diff_sum_sq = None
        self.diff_sum_sq_d = None
        self.prb_upd_d = None
        self.norm_obj_d = None
        self.kernel_chi_prb_obj = None 
        self.kernel_chi_sum_block = None 
        self.kernel_chi_reduce = None 
        self.kernel_dm_prb_obj = None 
        self.kernel_dm_cal_dev = None 
        self.kernel_dm_reduce_dev = None 
        self.kernel_dm_k4 = None 
        self.kernel_dm_k5 = None 
        self.kernel_prob_trans = None
        self.kernel_prob_reduce = None
        self.kernel_obj_trans = None
        self.plan_f = None
        self.last = None
        self.plan_last = None
        self.gpu_batch_size = 256
        
        
         
        #for timing
        self.elaps=[0.0]*20

        if self.sf_flag:
            self.use_scipy_fft()
            #self.use_pyfftw_fft()
        else:
            self.use_numpy_fft()

    def gpu_init(self):

        self.last = self.num_points % self.gpu_batch_size

        # Load arrays to GPU.
        self.diff_sum_sq = np.sum(self.diff_array**2, axis=(1,2))
        self.diff_sum_sq_d = gpuarray.to_gpu(self.diff_sum_sq)
        self.point_info_d  = gpuarray.to_gpu(np.int32(self.point_info))
        self.prb_d = gpuarray.to_gpu(self.prb) 
        self.obj_d = gpuarray.to_gpu(self.obj) 
        #self.prb_d = cuda.mem_alloc(self.prb.size * self.prb.dtype.itemsize)
        #self.obj_d = cuda.mem_alloc(self.obj.size * self.obj.dtype.itemsize)
        self.diff_d = gpuarray.to_gpu(self.diff_array)

        print "product type ", type(self.product) , np.shape(self.product)
        product=np.array(self.product)
        self.product_d = gpuarray.to_gpu(product)


        # complex temp buffs
        self.prb_obj_d = gpuarray.empty((self.gpu_batch_size,self.nx_prb,self.ny_prb),dtype=np.complex128)       
        self.fft_tmp_d = gpuarray.empty_like(self.prb_obj_d)
        self.prb_upd_d = gpuarray.empty_like(self.prb_d)

        #  float temp buffs
        self.amp_tmp_d = gpuarray.empty((self.gpu_batch_size,self.nx_prb,self.ny_prb), dtype=np.float64)
        self.dev_d = gpuarray.empty_like(self.amp_tmp_d)
        self.dev_buff_d = gpuarray.empty((self.gpu_batch_size,self.nx_prb),dtype=np.float64)
        self.power_d = gpuarray.empty((self.gpu_batch_size,), dtype=np.float64 )
        self.obj_norm_d = gpuarray.empty(np.shape(self.prb), dtype=np.float64 )
        self.prb_norm_d = gpuarray.empty(np.shape(self.obj), dtype=np.float64 )

        # make plan for cufft
        self.plan_f = cu_fft.Plan( np.shape(self.prb), np.complex128, np.complex128, self.gpu_batch_size)
        # last flan have different number of points.
        self.plan_last = cu_fft.Plan( np.shape(self.prb), np.complex128, np.complex128, self.last)




        func_mod = SourceModule("""
        #include <cuComplex.h>
        #include <math.h>
        #include <stdio.h>
        extern "C" {
        __global__ void cal_prb_obj(cuDoubleComplex *prb, cuDoubleComplex *obj, cuDoubleComplex *prb_obj, int * point_info, int  nx, int ny ,int o_ny, int offset  )
        {


        int i = blockIdx.x/nx ;
        int x = blockIdx.x % nx ;
        int xstart = point_info[(i+offset)*4];
        int ystart = point_info[(i+offset)*4+2];

	

        int tid = threadIdx.x ;

        int idx_p = tid + x* ny ;
        int idx_o = tid + (x  + xstart )*o_ny + ystart ;
        int idx_po = idx_p + i * nx * ny ;

        cuDoubleComplex result = cuCmul(prb[idx_p],obj[idx_o]) ;
        prb_obj[idx_po] = result ;


        }
        }



        extern "C" {
        __global__ void chi_sum_block( cuDoubleComplex * prb_obj, double * diff, double* diff_sum_sq , double* buff, int scale, int offset  )
        {
        extern  __shared__ double sdata[];

        
        unsigned int tid =threadIdx.x ;
        unsigned int idx = tid + blockDim.x * blockIdx.x ;
        unsigned int point = idx/scale + offset ;
        unsigned long idx_pro = idx + offset * scale ;
        double norm = diff_sum_sq[point] ;
        sdata[tid]=0.0 ;
        
        
        
        double chi = cuCabs(prb_obj[idx])/sqrt(double(scale))-diff[idx_pro] ;
        chi *= chi ;
        if (norm > 0.0) sdata[tid] = chi/norm ;
        __syncthreads() ;

        for (unsigned int s=blockDim.x/2; s>32; s>>=1)
        {
            if (tid < s) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        if (tid < 32)
        {
        sdata[tid] += sdata[tid + 32];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 16];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 8];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 4];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 2];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 1];
        }

        if(tid ==0 ) buff[blockIdx.x]=sdata[0] ; 
        //if(tid ==0 && blockIdx.x < 16) printf("buff[%d]=%f , %d \\n",blockIdx.x,sdata[0], point ) ;

        
        }
        }

        extern "C" {
        __global__ void chi_reduce( double* buff, double * buff1, int N )
        {
        extern __shared__ double sdata[];


        
        unsigned int tid = threadIdx.x ;
        sdata[0] = 0.0 ;
        unsigned int i = tid + blockDim.x * blockIdx.x  ;
        if ( i < N ) 
        sdata[tid] = buff[i];

        for (unsigned int s=blockDim.x/2; s>32; s>>=1)
        {
            if (tid < s)
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid < 32)
        {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
        }

        if (tid ==0 ) buff1[blockIdx.x]=sdata[0] ;


        }

        }

        extern "C" {
        __global__ void dm_prb_obj(cuDoubleComplex *prb, cuDoubleComplex *obj, cuDoubleComplex *prb_obj, cuDoubleComplex* product , cuDoubleComplex* tmp_fft, int * point_info, int  nx, int ny , int o_ny, int offset )
        {

        int i = blockIdx.x/nx ;
        int x = blockIdx.x % nx ;
        int xstart = point_info[(i+offset)*4];
        int ystart = point_info[(i+offset)*4+2];

        

        int tid = threadIdx.x ;

        int idx_p = tid + x* ny ;
        int idx_o = tid + (x  + xstart )*o_ny + ystart ;
        int idx_po = idx_p + i* nx * ny ;
        unsigned long idx_pro = idx_p + (i + offset) * nx * ny ;

        cuDoubleComplex result = cuCmul(prb[idx_p],obj[idx_o]) ;
        prb_obj[idx_po] = result ;
        tmp_fft[idx_po] =  cuCsub( cuCadd(result,result) , product[idx_pro] ) ; 
        


        }
        }


        extern "C" {
        __global__ void dm_cal_dev(cuDoubleComplex *fft_tmp, double * amp_tmp, double *diff, double* dev_d, double* dev_tmp , int  nx , double sigma1, int offset )
        {
        extern __shared__ double sdata[] ; 
        
        unsigned int tid = threadIdx.x ;
        unsigned long idx=tid+blockDim.x*blockIdx.x ;
        unsigned long idx_diff = idx + offset * blockDim.x * nx ;
        unsigned int i = blockIdx.x /nx ;
        double scale =  double(nx*blockDim.x) ;
        double scale_sqrt = sqrt(scale) ;
        cuDoubleComplex  fft = cuCmul(fft_tmp[idx], make_cuDoubleComplex(1.0/scale_sqrt,0.0))  ;
        double amp = cuCabs( fft );
        cuDoubleComplex ph = cuCmul(fft, make_cuDoubleComplex(1.0/(amp+sigma1),0.0) ) ;
        fft_tmp[idx] = ph ;
         
        
        amp_tmp[idx] = amp ;
        if( diff[idx_diff] >= 0.0 ) {
        double dev = amp - diff[idx_diff ] ;
        dev_d[idx] = dev ;
        sdata[tid] = dev * dev ;
        __syncthreads();

        for (unsigned int s=blockDim.x/2; s>32; s>>=1)
        {
            if (tid < s) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        if (tid < 32)
        {
        sdata[tid] += sdata[tid + 32];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 16];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 8];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 4];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 2];
            __syncthreads() ;
        }

        if ( tid == 0 ) 
        dev_tmp[blockIdx.x] = (sdata[0]+sdata[1])/scale ;
        
        }
        else 
        dev_tmp[blockIdx.x] = 0.0 ; 


        }
        }

        extern "C" {
        __global__ void dm_reduce_dev(double* dev_tmp,  double * power   )
        {
        extern __shared__ double sdata[] ; 

        
        unsigned int tid = threadIdx.x ;
        unsigned int idx = tid + blockDim.x * blockIdx.x  ;
        sdata[tid] = dev_tmp[idx];
        __syncthreads();

        for (unsigned int s=blockDim.x/2; s>32; s>>=1)
        {
            if (tid < s) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        if (tid < 32)
        {
        sdata[tid] += sdata[tid + 32];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 16];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 8];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 4];
            __syncthreads() ;
        sdata[tid] += sdata[tid + 2];
            __syncthreads() ;
        }

        if ( tid == 0 ) 
        power[blockIdx.x] =sdata[0] + sdata[1] ;


        }
        }

        
        extern "C" {
        __global__ void dm_k4(double* diff, double* dev, double * power, double* amp, cuDoubleComplex* fft_tmp, double sigma2, int nx, int offset  )
        {

        unsigned int tid = threadIdx.x ;
        unsigned int idx = tid + blockDim.x * blockIdx.x  ;
        unsigned long idx_diff = idx + offset * blockDim.x * nx ;
        int i = blockIdx.x /nx ;

        double amp_tmp = amp[idx] ;
        
        if( diff[idx_diff] >= 0.0 && power[i] > sigma2 ) 
            amp_tmp = diff[idx_diff] + dev[idx] * sqrt ( sigma2/power[i] ) ;
        
        fft_tmp[idx] = cuCmul(fft_tmp[idx], make_cuDoubleComplex(amp_tmp, 0.0) ) ; 
       

        }
        }
        
        extern "C" {
        __global__ void dm_k5(cuDoubleComplex* fft_tmp, cuDoubleComplex* product, cuDoubleComplex* obj_prb, int n,  double beta, double scale,int nx , int ny,int offset  )
        {

        unsigned int tid = threadIdx.x ;
        unsigned int idx = tid + blockDim.x * blockIdx.x  ;
        unsigned long idx_pro = idx + offset* nx * ny ;

        if( idx < n) { 
        cuDoubleComplex tmp = cuCmul(fft_tmp[idx],make_cuDoubleComplex(scale,0.0) );
        cuDoubleComplex tmp2 = cuCmul(cuCsub(tmp, obj_prb[idx]) , make_cuDoubleComplex(beta,0.0 ) ) ;
        product[idx_pro] = cuCadd(product[idx_pro], tmp2) ;
        }

        }
        }
        
        extern "C" {
        __global__ void prb_trans(cuDoubleComplex* product, cuDoubleComplex* obj, cuDoubleComplex * p_upd, double * norm_upd,  cuDoubleComplex* prb, double * norm, int * point_info, 
        int size, int nx, int ny, int o_ny ,int offset  )
        {
        __shared__ cuDoubleComplex p[16][17] ;
        __shared__ double n[16][17] ;

        unsigned int idx_pr = threadIdx.x + blockIdx.x* blockDim.x ;
        unsigned int x = idx_pr /ny ;
        unsigned int y = idx_pr % ny ;
        unsigned int i = threadIdx.y + blockIdx.y* blockDim.y ;
        int xstart = point_info[(i+offset)*4];
        int ystart = point_info[(i+offset)*4+2];
        
        unsigned int idx_o =  y + (x  + xstart )*o_ny + ystart ;
        unsigned long int idx_product = idx_pr + (i+offset) * nx * ny ;
        unsigned long idx_upd = idx_pr + blockIdx.y * nx * ny ;

        

        if( i < size &&  idx_pr < nx*ny ) { 
        cuDoubleComplex o = obj[idx_o] ;
        n[threadIdx.x][threadIdx.y] = cuCabs(o) * cuCabs(o) ; 
        p[threadIdx.x][threadIdx.y] = cuCmul(product[idx_product], cuConj(o)  );
        } 
        else { 
        p[threadIdx.x][threadIdx.y] = make_cuDoubleComplex(0.0, 0.0) ;
        n[threadIdx.x][threadIdx.y] =0.0 ;
        }
        __syncthreads() ;
        
        for (unsigned int s=blockDim.y/2; s>1; s>>=1)
        {
            if (threadIdx.y  < s) {
            p[threadIdx.x][threadIdx.y] = cuCadd( p[threadIdx.x][threadIdx.y], p[threadIdx.x][threadIdx.y + s]) ;
            n[threadIdx.x][threadIdx.y] += n[threadIdx.x][threadIdx.y + s] ;  }
            __syncthreads();
        }
        if (threadIdx.y == 0 && idx_pr < nx*ny  && gridDim.y > 1 )  {
            p_upd[idx_upd] = cuCadd (p[threadIdx.x][0], p[threadIdx.x][1] ) ; 
            norm_upd[idx_upd] = n[threadIdx.x][0] + n[threadIdx.x][1] ;
        } 
        else if (threadIdx.y == 0 && idx_pr < nx*ny ) {
            prb[idx_pr] = cuCadd(prb[idx_pr],  cuCadd (p[threadIdx.x][0], p[threadIdx.x][1] )) ; 
            norm[idx_pr] = norm[idx_pr] + n[threadIdx.x][0] + n[threadIdx.x][1] ;
        }    

        }
        }
        
        extern "C" {
        __global__ void prb_reduce(cuDoubleComplex* prb_upd, double * norm_upd,  cuDoubleComplex *  prb, double * norm,  int size, int prb_sz ,int offset  )
        {
        __shared__ cuDoubleComplex p[16][17]  ;
        __shared__ double n[16][17] ;

        unsigned int idx_pr = threadIdx.x + blockIdx.x* blockDim.x ;
        unsigned int i = threadIdx.y + blockIdx.y* blockDim.y ;
        unsigned long  idx_upd = idx_pr + i * prb_sz  ;

        if ( idx_pr < prb_sz ) {
        p[threadIdx.x][threadIdx.y] = prb_upd[idx_upd]  ;
        n[threadIdx.x][threadIdx.y] = norm_upd[idx_upd] ;
        }
        else {
        p[threadIdx.x][threadIdx.y] = make_cuDoubleComplex(0.0, 0.0) ;
        n[threadIdx.x][threadIdx.y] =0.0 ;
        }
        

        __syncthreads() ;

        for (unsigned int s=blockDim.y/2; s>1; s>>=1)
        {
            if (threadIdx.y  < s) {
            p[threadIdx.x][threadIdx.y] = cuCadd( p[threadIdx.x][threadIdx.y], p[threadIdx.x][threadIdx.y + s]) ;
            n[threadIdx.x][threadIdx.y] += n[threadIdx.x][threadIdx.y + s] ;  }
            __syncthreads();
        }
        if (threadIdx.y == 0 && idx_pr < prb_sz  )  {
            if (offset != 0 ) {
                prb[idx_pr] = cuCadd( prb[idx_pr], cuCadd (p[threadIdx.x][0], p[threadIdx.x][1] ) );
                norm[idx_pr] = norm[idx_pr] + n[threadIdx.x][0] + n[threadIdx.x][1] ;
                } else {
                prb[idx_pr] = cuCadd (p[threadIdx.x][0], p[threadIdx.x][1] ) ;
                norm[idx_pr] = n[threadIdx.x][0] + n[threadIdx.x][1] ;
            }
        }
        
        }
        }
        
        extern "C" {
        __global__ void prb_trans(cuDoubleComplex* product, cuDoubleComplex* obj, cuDoubleComplex * prb,   double * norm, int * point_info, 
         double alpha, int o_ny , int i  )
        {
        idx_pr = treadIdx.x + blockIdx.x * blockDim.x ;
        idx_pd = idx_pr + i* blockDim.x*gridDim.x ;

        int xstart = point_info[i*4];
        int ystart = point_info[i*4+2];
        
        unsigned int idx_o =  threadIdx.x + (blockIdx.x  + xstart )*o_ny + ystart ;
        cuDoubleComplex p = prb[idx_pr] ;
        double p2 = cuCabs(p) * cuCabs(p) ;

        if( i ==0 ) {
            obj[idx_o] = cuCmul(cuConj(p) , product[idx_pd] ) ;
            norm[idx_o] = alpha + p2 ;
        }  else {
        obj[idx_o] = cuCadd(obj[idx_o] , cuCmul(cuConj(p) , product[idx_pd] ))
        norm[idx_o] = norm[idx_o] + p2 ; 
        }


        }
        }
    


        """, no_extern_c=1)

        self.kernel_chi_prb_obj = func_mod.get_function("cal_prb_obj") 
        self.kernel_chi_sum_block = func_mod.get_function("chi_sum_block") 
        self.kernel_chi_reduce = func_mod.get_function("chi_reduce") 
        self.kernel_chi_reduce = func_mod.get_function("chi_reduce") 
        self.kernel_dm_prb_obj = func_mod.get_function("dm_prb_obj") 
        self.kernel_dm_cal_dev = func_mod.get_function("dm_cal_dev") 
        self.kernel_dm_reduce_dev = func_mod.get_function("dm_reduce_dev") 
        self.kernel_dm_k4 = func_mod.get_function("dm_k4") 
        self.kernel_dm_k5 = func_mod.get_function("dm_k5") 
        self.kernel_prob_trans = func_mod.get_function("prb_trans") 
        self.kernel_prob_reduce = func_mod.get_function("prb_reduce") 
        self.kernel_obj_trans = func_mod.get_function("obj_trans"") 

    def use_pyfftw_fft(self):
        global ifftshift
        global fftshift
        global ifftn
        global fftn

        print('Using pyfftw')
        ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
        fftshift = pyfftw.interfaces.numpy_fft.fftshift
        fftn = pyfftw.interfaces.numpy_fft.fftn
        ifftn = pyfftw.interfaces.numpy_fft.ifftn

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)

    def use_scipy_fft(self):
        global ifftshift
        global fftshift
        global ifftn
        global fftn

        print('Using scipy fft')
        ifftshift = sf.ifftshift
        fftshift = sf.fftshift
        ifftn = sf.ifftn
        fftn = sf.fftn

    def use_numpy_fft(self):
        global ifftshift
        global fftshift
        global ifftn
        global fftn

        print('Using numpy fft')
        ifftshift = np.fft.ifftshift
        fftshift = np.fft.fftshift
        ifftn = np.fft.ifftn
        fftn = np.fft.fftn

    # create array with pixel value equals euclidian distance from array center
    def dist(self):
        a = np.arange(self.kernal_n)
        a = np.where(a < np.float(self.kernal_n)/2., a, np.abs(a-np.float(self.kernal_n)))**2
        array = np.zeros((self.kernal_n, self.kernal_n))
        for i in range(np.int(self.kernal_n)//2+1):
            y = np.sqrt(a+i**2)
            array[:, i] = y
            if i != 0:
                array[:, self.kernal_n-i] = y
        return fftshift(array)

    def dist_n(self,dims):
        a = np.arange(dims)
        a = np.where(a<np.float(dims)/2.,a,np.abs(a-np.float(dims)))**2
        array=np.zeros((dims,dims))
        for i in range(np.int(dims//2)+1):
            y=np.sqrt(a+i**2)
            array[:,i]=y
            if i!=0:
                array[:,dims-i]=y
        del(a)
        del(y)
        return np.fft.fftshift(array)

    # generate Gaussian function
    def create_gauss(self,dims,sigma):
        rr = self.dist_n(dims)
        gf = np.exp(-1.*(rr/sigma)**2/2.)
        norm = np.sum(gf)
        return gf/norm

    def create_gauss_2d(self,s):
        rr = self.dist_n(self.nx_prb)
        gf = np.exp(-1.*(rr/s)**2/2.)
        norm = np.sum(gf)
        return gf/norm

    def congrid(self, array_in, shape):
        x_in,y_in = np.shape(array_in)
        x = np.arange(x_in)
        y = np.arange(y_in)

        kernel = interpolate.RectBivariateSpline(x,y,array_in, kx=2,ky=2)
        xx = np.linspace(x.min(),x.max(),shape[0])
        yy = np.linspace(y.min(),y.max(),shape[1])

        return  kernel(xx,yy)

    def congrid_fft(self,array_in, shape):
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

    # generate Tukey window filter function
    def create_tukey(self,dims,alpha=0.1):
        tukey_filter = np.zeros(dims)
        for i in range(np.int(dims[0])):
            if(i <= (alpha*(dims[0]-1)/2)):
                x_factor = 0.5*(1+np.cos(np.pi*(2*i/(alpha*(dims[0]-1))-1)))
            elif (i >=((dims[0]-1)*(1-alpha/2))):
                x_factor = 0.5*(1+np.cos(np.pi*(2*i/(alpha*(dims[0]-1))-2/alpha+1)))
            else:
                x_factor = 1
            for j in range(np.int(dims[1])):
                if(j <= (alpha*(dims[1]-1)/2)):
                    y_factor = 0.5*(1+np.cos(np.pi*(2*j/(alpha*(dims[1]-1))-1)))
                elif (j >=((dims[1]-1)*(1-alpha/2))):
                    y_factor = 0.5*(1+np.cos(np.pi*(2*j/(alpha*(dims[1]-1))-2/alpha+1)))
                else:
                    y_factor = 1
                tukey_filter[i,j] = x_factor * y_factor
        return tukey_filter

    # rebin array size
    def rebin(self, a, *args,**kwargs):
        shape = a.shape
        lenShape = a.ndim
        factor = np.asarray(shape)/np.asarray(args)
        evList = ['a.reshape('] + \
            ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
            [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
            ['*( 1.'] + ['/factor[%d]'%i for i in range(lenShape)] + [')']
        return eval(''.join(evList))

    # Lucy-Richardson deconvolution
    def RL_deconv(self,image,PSF, iterations):
        #PSF /= np.sum(PSF)
        latent_est = image
        PSF_HAT = np.flipud(np.fliplr(PSF))
        for i in range(iterations):
            est_conv = convolve2d(latent_est, PSF, 'same')
            relative_blur = image / est_conv
            error_est = convolve2d(relative_blur, PSF_HAT, 'same')
            latent_est = latent_est * error_est
        return latent_est

    # Wiener filter
    def wiener_filter(self,img, kernel, K = 0.01):
        dummy = np.copy(img)
        kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
        dummy = np.fft.fftn(dummy)
        kernel = np.fft.fftn(kernel)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        dummy = dummy * kernel
        dummy = np.abs(np.fft.ifftn(dummy))
        return dummy


    # calculate convolution
    def cal_convol(self):
        nx, ny = np.shape(self.conv_array1)
        FT1 = ifftshift(ifftn(fftshift(self.conv_array1))) * np.sqrt(1.*nx*ny)

        if self.conv_flag:
            tmp = np.zeros((nx, ny))
            nx_2, ny_2 = np.shape(self.conv_array2)
            tmp[nx//2-nx_2//2:nx//2+nx_2//2, ny//2-ny_2//2:ny//2+ny_2//2] = self.conv_array2
            FT2 = ifftshift(ifftn(fftshift(tmp)))*np.sqrt(1.*nx*ny)
        else:
            FT2 = ifftshift(ifftn(fftshift(self.conv_array2)))*np.sqrt(1.*nx*ny)

        if self.conv_complex_flag:
            if self.conv_norm_flag:
                return ifftshift(fftn(fftshift(FT1*FT2)))
            else:
                return ifftshift(fftn(fftshift(FT1*FT2)))/np.sqrt(1.*nx*ny)
        else:
            if self.conv_norm_flag:
                return np.abs((ifftshift(fftn(fftshift(FT1*FT2)))))
            else:
                return np.abs(ifftshift(fftn(fftshift(FT1*FT2)))/np.sqrt(1.*nx*ny))

    # calculate round roi scan parttern
    def cal_scan_pattern(self):

        x_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um)
        y_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um)

        r_max_um = np.sqrt((self.x_range_um/2.)**2+(self.y_range_um/2.)**2)
        num_ring = 1+int(r_max_um / self.dr_um)

        self.points = np.zeros((2, self.num_points))
        # self.points = np.zeros((2,1000))
        i_positions = 0
        for i_ring in range(1, num_ring+2):
            radius_um = i_ring * self.dr_um
            angle_step = 2. * math.pi / (i_ring * self.nth)
            for i_angle in range(int(i_ring * self.nth)):
                angle = i_angle * angle_step
                x_um = radius_um * np.cos(angle)
                y_um = radius_um * np.sin(angle)
                if abs(x_um) <= (self.x_range_um/2):
                    if abs(y_um) <= (self.y_range_um/2):
                        self.points[0, i_positions] = self.x_direction * np.round(-1.*x_um * 1.e3 / x_real_space_pixel_nm)
                        self.points[1, i_positions] = self.y_direction * np.round(-1.*y_um * 1.e3 / y_real_space_pixel_nm)
                        if self.bragg_flag:
                            self.points[0, i_positions] = np.round(self.points[0, i_positions] * np.cos(self.bragg_delta*3.14/180.))
                            self.points[1, i_positions] = np.round(self.points[1, i_positions] * np.cos(self.bragg_gamma*3.14/180.))
                        i_positions = i_positions + 1

    # convert scan pattern into detector frame
    def convert_scan_pattern(self):
        theta = self.bragg_theta * np.pi / 180.
        gamma = self.bragg_gamma * np.pi / 180.
        delta = self.bragg_delta * np.pi / 180.
        conv_matrix = np.asarray([[(np.cos(gamma) * np.cos(theta) - np.sin(gamma) * np.sin(theta)),0],\
                                      [(np.sin(delta)*np.sin(gamma)*np.cos(theta)+np.cos(gamma)*np.sin(delta)*np.sin(theta)),np.cos(delta)]])
        for i in range(self.num_points):
            x_tmp = self.points[0,i] * conv_matrix[0,0] + self.points[1,i] * conv_matrix[0,1]
            y_tmp = self.points[0,i] * conv_matrix[1,0] + self.points[1,i] * conv_matrix[1,1]
            self.points[0,i] = x_tmp
            self.points[1,i] = y_tmp

    # calculate mesh scan parttern
    def cal_scan_pattern_mesh(self):

        x_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um)
        y_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um)

        self.n_col = np.int(np.round(self.x_range_um / self.x_dr_um)) + 1
        self.n_row = np.int(np.round(self.y_range_um / self.y_dr_um)) + 1
        print(self.x_range_um,self.x_dr_um,self.n_col)
        print(self.y_range_um,self.y_dr_um,self.n_row)
        self.points = np.zeros((2, self.n_col*self.n_row))
        i_positions = 0
        for iy in range(self.n_row):
            for ix in range(self.n_col):
                if self.bragg_flag:
                    theta = self.bragg_theta * np.pi / 180.
                    gamma = self.bragg_gamma * np.pi / 180.
                    delta = self.bragg_delta * np.pi / 180.
                    conv_matrix = np.asarray([[(np.cos(gamma) * np.cos(theta) - np.sin(gamma) * np.sin(theta)),0],\
                                                  [(np.sin(delta)*np.sin(gamma)*np.cos(theta)+np.cos(gamma)*np.sin(delta)*np.sin(theta)),np.cos(delta)]])
                    x_tmp = self.x_direction * (-1*(ix - self.n_col/2)) \
                        * self.x_dr_um * 1.e3 / x_real_space_pixel_nm
                    y_tmp = self.y_direction * (-1*(iy - self.n_row/2)) \
                        * self.y_dr_um * 1.e3 / y_real_space_pixel_nm
                    self.points[0,i_positions] = np.round(x_tmp * conv_matrix[0,0] + y_tmp * conv_matrix[0,1])
                    self.points[1,i_positions] = np.round(x_tmp * conv_matrix[1,0] + y_tmp * conv_matrix[1,1])
                else:
                    self.points[0, i_positions] = self.x_direction * np.round(-1*(ix - self.n_col/2) * self.x_dr_um * 1.e3 / x_real_space_pixel_nm)
                    self.points[1, i_positions] = self.y_direction * np.round(-1*(iy - self.n_row/2) * self.y_dr_um * 1.e3 / y_real_space_pixel_nm)
                i_positions += 1

    def cal_scan_pattern_fermat(self):

        x_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um)
        y_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um)

        r_max_um = np.sqrt((self.x_range_um/2.)**2+(self.y_range_um/2.)**2)
        r_range = np.int((1.5*r_max_um/self.dr_um)**2)

        points_tmp = np.zeros((2, 100000))
        i_positions = 0
        i_positions_2 = 1
        for i_ring in range(1, r_range):
            radius_um = np.sqrt(i_positions_2) * self.dr_um
            angle = i_positions_2 * 137.508 * math.pi / 180.
            x_um = radius_um * np.cos(angle)
            y_um = radius_um * np.sin(angle)
            i_positions_2 = i_positions_2 + 1
            if abs(x_um) <= (self.x_range_um/2):
                if abs(y_um) <= (self.y_range_um/2):
                    if self.bragg_flag:
                        #print('cal bragg points ', self.bragg_theta, self.bragg_gamma,self.bragg_delta)
                        theta = self.bragg_theta * np.pi / 180.
                        gamma = self.bragg_gamma * np.pi / 180.
                        delta = self.bragg_delta * np.pi / 180.
                        conv_matrix = np.asarray([[(np.cos(gamma) * np.cos(theta) - np.sin(gamma) * np.sin(theta)),0],\
                                                  [(np.sin(delta)*np.sin(gamma)*np.cos(theta)+np.cos(gamma)*np.sin(delta)*np.sin(theta)),np.cos(delta)]])
                        x_tmp = self.x_direction * (-1.*x_um * 1.e3 / x_real_space_pixel_nm)
                        y_tmp = self.y_direction * (-1.*y_um * 1.e3 / y_real_space_pixel_nm)
                        points_tmp[0, i_positions] = np.round(x_tmp * conv_matrix[0,0] + y_tmp * conv_matrix[0,1])
                        points_tmp[1, i_positions] = np.round(x_tmp * conv_matrix[1,0] + y_tmp * conv_matrix[1,1])
                    else:
                        points_tmp[0, i_positions] = self.x_direction * np.round(-1.*x_um * 1.e3 / x_real_space_pixel_nm)
                        points_tmp[1, i_positions] = self.y_direction * np.round(-1.*y_um * 1.e3 / y_real_space_pixel_nm)
                    i_positions = i_positions + 1

        self.points = points_tmp[:,:i_positions].copy()
        del(points_tmp)

    def recon_dm_trans_pc(self):
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
            tmp = 2. * prb_obj - self.product[i]
            if np.sum(self.diff_array[i]) > 0.:
                tmp_fft = fftn(tmp) / np.sqrt(np.size(tmp))
                I_c = np.abs(np.fft.fftshift(tmp_fft))**2
                I_pc = np.fft.fftshift(np.roll(np.roll(convolve2d(I_c,self.coh,'same'),-1,0),-1,1))
                amp_tmp = np.abs(tmp_fft)
                ph_tmp = tmp_fft / (amp_tmp+self.sigma1)
                (index_x, index_y) = np.where(self.diff_array[i] >= 0.)
                dev = np.sqrt(I_pc) - self.diff_array[i]
                power = np.sum((dev[index_x, index_y])**2)/(self.nx_prb*self.ny_prb)
                if power > self.sigma2:
                    amp_tmp[index_x, index_y] = self.diff_array[i][index_x, index_y] + dev[index_x, index_y] * np.sqrt(self.sigma2/power)
                tmp2 = ifftn(amp_tmp*ph_tmp) *  np.sqrt(np.size(tmp))
                self.product[i] += self.beta*(tmp2 - prb_obj)
            else:
                self.product[i] = tmp

    # recover coherence function with diffraction center array
    def cal_coh(self, it_count):
        if((it_count >= (self.pc_start * self.n_iterations)) & (it_count <= (self.pc_end * self.n_iterations))):
            if np.mod(it_count,self.pc_step) == 0:
                self.coh_old = self.coh.copy()
                coh = np.zeros_like(self.coh)
                coh_num_points = np.int(self.coh_percent*self.num_points)

                for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info[:coh_num_points,:]):
                    if np.sum(self.diffint_center[i,:,:] > 0):
                        tmp = self.prb * self.obj[x_start:x_end, y_start:y_end]
                        tmp_fft = np.abs(ifftshift(fftn(fftshift(tmp))) / np.sqrt(self.nx_prb * self.ny_prb * 1.))**2
                        if self.pc_kernel_n == self.nx_prb:
                            int_c = tmp_fft * self.pc_filter
                        else:
                            int_c = self.rebin(tmp_fft[self.nx_prb//2-self.pc_kernel_n:self.nx_prb//2+self.pc_kernel_n,\
                                                       self.ny_prb//2-self.pc_kernel_n:self.ny_prb//2+self.pc_kernel_n],\
                                               self.pc_kernel_n,self.pc_kernel_n) * self.pc_filter
                        int_pc = self.diffint_center[i,:,:] * self.pc_filter
                        if self.pc_alg == 'lucy':
                            coh += self.RL_deconv(int_pc,int_c,self.lucy_it_num)
                        elif self.pc_alg == 'wiener':
                            coh += np.fft.fftshift(self.wiener_filter(int_pc,int_c,self.pc_wiener_factor))
                coh /= coh_num_points
                self.coh = coh / np.sum(coh)
                mc_nx, mc_ny = self.cal_mass_center(self.coh)
                self.coh = np.roll(np.roll(self.coh,np.int(self.pc_kernel_n//2-mc_nx),0),np.int(self.pc_kernel_n//2-mc_ny),1)
                error_coh = np.sum(np.abs(self.coh - self.coh_old)**2) / np.sum(np.abs(self.coh)**2)
                print('coh updated, coh_chi = ', error_coh)

    # difference map for ptychography
    def recon_dm_trans_real(self):

        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
            tmp = 2. * prb_obj - self.product[i]
            diff = self.diff_array[i]

            if np.sum(diff) > 0.:
                tmp_fft = fftn(tmp) / np.sqrt(np.size(tmp))
                index = np.where(diff > 0)
                tmp_fft[index] = diff[index] * np.exp(1j*np.angle(tmp_fft[index]))
                tmp2 = ifftn(tmp_fft) *  np.sqrt(np.size(tmp))
                self.product[i] += self.beta*(tmp2 - prb_obj)
                #self.product[i] += (tmp2 - prb_obj)
            else:
                self.product[i] = prb_obj

    # error reduction for ptychography
    def recon_er_trans(self):

        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
            diff = self.diff_array[i]

            if np.sum(diff) > 0.:
                tmp_fft = fftn(prb_obj) / np.sqrt(np.size(prb_obj))
                tmp_fft[diff > 0.] = diff[diff > 0.] * np.exp(1j*np.angle(tmp_fft[diff > 0.]))

                self.product[i] = ifftn(tmp_fft) *  np.sqrt(np.size(tmp_fft))

            else:
                self.product[i] = prb_obj

    # difference map for ptychography
    def recon_ml_trans(self):
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
            diff = self.diff_array[i]

            if np.sum(diff) > 0.:
                diff_int = diff**2
                tmp_fft = fftn(prb_obj) / np.sqrt(np.size(prb_obj))
                if self.ml_mode == 'Gaussian':
                    a = -3.*(diff_int + 1.e-9) + 0.j
                    b = 2. * diff_int - 0.5 * self.ml_weight * np.abs(tmp_fft) * diff - self.ml_weight + 0.j
                    c = -1 * self.ml_weight * np.abs(tmp_fft) * diff + self.ml_weight * diff_int + 0.j
                    sol = (-1*b + np.sqrt(b**2-4.*a*c)) / (2.*a)
                    v = diff * np.sqrt(1.+ sol)
                    v_complex = v * np.exp(1j*np.angle(tmp_fft))
                    #v_complex[diff == 0.] = tmp_fft[diff == 0.]
                    self.product[i] = ifftn(v_complex) * np.sqrt(np.size(v_complex))
                elif self.ml_mode == 'Poisson':
                    '''
                    b = 1./self.ml_weight - np.abs(tmp_fft) + 0.j
                    c = -1 * diff_int / self.ml_weight + 0.j
                    a = (-1*b**3+1.5*(-9.0*c + np.sqrt(3.0)*np.sqrt(c*(4.0*b**3+27.0*c))))**(1/3.0)
                    sol = (1.0/3.0)*(a + b*(-1.0 + b/a))
                    '''
                    maxLH = diff_int - diff_int*np.log(diff_int+1.e-8)
                    LH = np.abs(tmp_fft)**2 - diff_int * np.log(np.abs(tmp_fft)**2+1.e-8) - maxLH
                    residual = (diff - np.abs(tmp_fft))**2
                    w = LH / residual
                    sol = (-w + np.sqrt(w**2 + 4.*(diff_int + w * np.abs(tmp_fft)))) / 2.
                    v_complex = np.real(sol) * np.exp(1j*np.angle(tmp_fft))
                    #v_complex[diff == 0.] = tmp_fft[diff == 0.]
                    self.product[i] = ifftn(v_complex) * np.sqrt(np.size(v_complex))
            else:
                self.product[i] = prb_obj

    # difference map for ptychography
    def recon_dm_trans_single(self, i):

        x_start, x_end, y_start, y_end = self.point_info[i]
        diff = self.diff_array[i]

        prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
        tmp = 2. * prb_obj - self.product[i]

        if np.sum(diff) > 0.:

            tmp_fft = fftn(tmp) / np.sqrt(np.size(tmp))

            amp_tmp = np.abs(tmp_fft)
            ph_tmp = tmp_fft / (amp_tmp + self.sigma1)

            index_x, index_y = np.where(diff >= 0.)
            dev = amp_tmp - diff
            power = np.sum((dev[index_x, index_y]) ** 2) / (self.nx_prb * self.ny_prb)

            if power > self.sigma2:
                a = diff + dev * np.sqrt(self.sigma2 / power)
                amp_tmp[index_x, index_y] = a[index_x, index_y]

            tmp2 = ifftn(amp_tmp * ph_tmp) * np.sqrt(np.size(tmp))

            return self.beta * (tmp2 - prb_obj)

        else:
            return tmp

    def recon_dm_trans(self):
        results = self._run_parallel_over_points(self.recon_dm_trans_single)

        for product, result in zip(self.product, results):
            product += result

    def dm_gpu_single(self, start_point, size ):
        #decide kernel block size to use
        #currently use  ny_prb
        block_size = self.ny_prb
        n_blocks = size*self.nx_prb
        #need to make is into numpy array to pass as argument to pycuda kernel launcher
        nx = np.int32(self.nx_prb)
        ny = np.int32(self.ny_prb)
        o_ny = np.int32(self.ny_obj)
        offset = np.int32(start_point)

        #launch kernel calculate obj*prb store in prb_obj_d and 2PO-Psi in fft_tmp_d
        self.kernel_dm_prb_obj(self.prb_d, self.obj_d, self.prb_obj_d, self.product_d, self.fft_tmp_d, self.point_info_d, \
                nx, ny, o_ny, offset, \
                block=(block_size,1,1), grid=(n_blocks,1,1) )

        #do in space fft on 2PO-Psi
        cu_fft.fft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f )

        #second kernel  store dev =amp_tmp-diff reduce sum(dev**2) in blocks.
        block_size = self.ny_prb 
        n_blocks = size * self.nx_prb 
        sigma1 = np.float64(self.sigma1) 
        self.kernel_dm_cal_dev(self.fft_tmp_d, self.amp_tmp_d, self.diff_d, self.dev_d,  self.dev_buff_d, nx ,  \
                sigma1, offset, \
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) , shared = self.ny_prb* 8 ) 
		
        # cpu calculate power or futher gpu reduce if points are a lot .
        block_size = self.nx_prb
        n_blocks = size
        self.kernel_dm_reduce_dev(self.dev_buff_d, self.power_d ,
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) , shared = self.nx_prb* 8 ) 
        
        # calculate diff+dev*sqrt(sigma2/power) 
        block_size = self.ny_prb
        n_blocks = size*self.nx_prb
        sigma2=np.float64(self.sigma2)
        self.kernel_dm_k4(self.diff_d, self.dev_d, self.power_d, self.amp_tmp_d , self.fft_tmp_d, sigma2, nx, offset, \
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) )

        # inverse fft
        cu_fft.ifft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f, True )

        # final update \psi
        beta = np.float64(self.beta) 
        scale = np.float64(np.sqrt(self.nx_prb * self.ny_prb)) 
        n = np.int32(size * self.nx_prb * self.ny_prb)
        block_size =128
        n_blocks = (n -1 )/block_size +1 
        self.kernel_dm_k5(self.fft_tmp_d, self.product_d, self.prb_obj_d,  \
                n, beta, scale, nx, ny, offset ,\
                block=( block_size, 1,1) , grid=(n_blocks, 1,1 ) )


    def recon_dm_trans_gpu(self ) :
        #load Obj and Prb to GPU, this is not needed if O and P update is done in GPU.
        #cuda.memcpy_htod(self.prb_d, self.prb )
        #cuda.memcpy_htod(self.obj_d, self.obj )

        n_batch = self.num_points/self.gpu_batch_size
        for i in range(n_batch) :
            self.dm_gpu_single( i * self.gpu_batch_size, self.gpu_batch_size)

        #lasr batch have different size :
        if self.last > 0 :
            self.dm_gpu_single( n_batch * self.gpu_batch_size , self.last ) 

        # this is not needed when O,P update is calculated in GPU
        self.product=self.product_d.get()

    def recon_dm_trans_gpu_save(self):

        

        t0=time.time()
        #load Obj and Prb to GPU, this is not needed if O and P update is done in GPU.
        cuda.memcpy_htod(self.prb_d, self.prb )
        cuda.memcpy_htod(self.obj_d, self.obj )
 
        #load product to GPU, this is not needed if all calculate is in GPU.
        #product=np.array(self.product)
        #self.product_d.set(product)

        cuda.Context.synchronize()
		
        t1=time.time()
        self.elaps[13] += t1-t0


        #decide kernel block size to use 
        #currently use  ny_prb
        block_size = self.ny_prb

        #number of blocks (grid size) for kernel launch
        n_blocks = (self.num_points*self.nx_prb*self.ny_prb-1)/block_size +1

        #need to make is into numpy array to pass as argument to pycuda kernel launcher
        nx = np.int32(self.nx_prb)
        ny = np.int32(self.ny_prb)
        o_ny = np.int32(self.ny_obj)
        points = np.int32(self.num_points)

        #launch kernel calculate obj*prb store in prb_obj_d and 2PO-Psi in fft_tmp_d
        self.kernel_dm_prb_obj(self.prb_d, self.obj_d, self.prb_obj_d, self.product_d, self.fft_tmp_d, self.point_info_d, \
                nx, ny, o_ny, points, \
                block=(block_size,1,1), grid=(n_blocks,1,1) )


        cuda.Context.synchronize()
		
        t0=time.time()
        self.elaps[14] += t0-t1

        
        #do in space fft on 2PO-Psi
        cu_fft.fft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f )
	

        cuda.Context.synchronize()
		
        t1=time.time()
        self.elaps[15] += t1-t0



        #second kernel  store dev =amp_tmp-diff reduce sum(dev**2) in blocks.
        
        block_size = self.ny_prb 
        n_blocks = self.num_points * self.nx_prb 
        sigma1 = np.float64(self.sigma1) 
        self.kernel_dm_cal_dev(self.fft_tmp_d, self.amp_tmp_d, self.diff_d, self.dev_d,  self.dev_buff_d, nx ,  \
                sigma1,
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) , shared = self.ny_prb* 8 ) 
		
        cuda.Context.synchronize()
        t0=time.time()
        self.elaps[16] += t0-t1

        # cpu calculate power or futher gpu reduce if points are a lot .

        block_size = self.nx_prb
        n_blocks = self.num_points

        self.kernel_dm_reduce_dev(self.dev_buff_d, self.power_d ,
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) , shared = self.nx_prb* 8 ) 


        #fft_tmp=self.fft_tmp_d.get()
        #ph_tmp=self.fft_tmp_d.get()
        #prb_obj=self.prb_obj_d.get() 
        #amp_tmp_a = self.amp_tmp_d.get() 
        
        #dev_buff=self.dev_buff_d.get()

        # calculate diff+dev*sqrt(sigma2/power) 
        block_size = self.ny_prb
        n_blocks = self.num_points*self.nx_prb
        sigma2=np.float64(self.sigma2)
        
        self.kernel_dm_k4(self.diff_d, self.dev_d, self.power_d, self.amp_tmp_d , self.fft_tmp_d, sigma2, nx, \
                block=(block_size, 1,1 ) , grid = ( n_blocks, 1,1) )



        # inverse fft
        cu_fft.ifft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f, True )

        
        # final update \psi

        beta = np.float64(self.beta) 
        scale = np.float64(np.sqrt(self.nx_prb * self.ny_prb)) 
        n = np.int32(self.num_points * self.nx_prb * self.ny_prb)
        block_size =128
        n_blocks = (n -1 )/block_size +1 
        self.kernel_dm_k5(self.fft_tmp_d, self.product_d, self.prb_obj_d,  \
                n, beta, scale, \
                block=( block_size, 1,1) , grid=(n_blocks, 1,1 ) )
        


        #fft_tmp=self.fft_tmp_d.get()
        #prb_obj=self.prb_obj_d.get() 

        #power = np.sum(dev_buff,1 ) 
        #power = self.power_d.get() 

        cuda.Context.synchronize()
		
        t1=time.time()
        self.elaps[17] += t1-t0

        self.product=self.product_d.get()
        '''
        for i in range(self.num_points):
            
            #x_start, x_end, y_start, y_end = self.point_info[i]
            diff = self.diff_array[i]

            if np.sum(diff) > 0.:

                #tmp2 = ifftn(fft_tmp[i]) * np.sqrt(self.nx_prb*self.ny_prb)
                tmp2 = fft_tmp[i] * np.sqrt(self.nx_prb*self.ny_prb)
                #tmp_fft = fft_tmp[i] / np.sqrt(self.nx_prb*self.ny_prb)

                #amp_tmp = np.abs(tmp_fft)
                #ph_tmp = tmp_fft / (amp_tmp + self.sigma1)
                #amp_tmp = amp_tmp_a[i]
                #index_x, index_y = np.where(diff >= 0.)
                #dev = amp_tmp - diff
                #power = np.sum((dev[index_x, index_y]) ** 2) / (self.nx_prb * self.ny_prb)

                #if power[i] > self.sigma2:
                #   a = diff + dev * np.sqrt(self.sigma2 / power[i])
                #  amp_tmp[index_x, index_y] = a[index_x, index_y]

                #tmp2 = ifftn(amp_tmp * ph_tmp[i]) * np.sqrt(self.nx_prb*self.ny_prb)
                self.product[i] += self.beta * (tmp2 - prb_obj[i])
            else:
                self.product[i] = prb_obj[i]
	'''	
        t0=time.time()
        self.elaps[18] += t0-t1


        #kernel prepare ifft 


        # kernel calculate psi += beta * (psi-P.O)

 
        


    # difference map for multislice case, only updates exitwaves on the last plane
    def recon_dm_trans_ms(self):
        print('update with diff pattern...')
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            diff = self.diff_array[i]

            prb_obj = self.prb_ms[i][self.slice_num-1] * self.obj_ms[x_start:x_end, y_start:y_end,self.slice_num-1]
            tmp = 2. * prb_obj - self.product[i][self.slice_num-1]

            tmp_fft = fftn(tmp) / np.sqrt(np.size(tmp))

            amp_tmp = np.abs(tmp_fft)
            ph_tmp = tmp_fft / (amp_tmp + self.sigma1)

            index_x, index_y = np.where(diff >= 0.)
            dev = amp_tmp - diff
            power = np.sum((dev[index_x, index_y]) ** 2) / (self.nx_prb * self.ny_prb)

            if power > self.sigma2:
                a = diff + dev * np.sqrt(self.sigma2 / power)
                amp_tmp[index_x, index_y] = a[index_x, index_y]

            tmp2 = ifftn(amp_tmp * ph_tmp) * np.sqrt(np.size(tmp))
            self.product[i][self.slice_num-1] = tmp2.copy()

    def recon_dm_trans_ms_bp(self):
        print('update with diff pattern...')
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            diff = self.diff_array[i]

            prb_obj = self.prb_ms[i][self.slice_num-1] * self.obj_ms[x_start:x_end, y_start:y_end,self.slice_num-1]
            tmp_fft = fftn(prb_obj) / np.sqrt(np.size(prb_obj))

            index = np.where(diff > 0.)
            tmp_fft[index] = tmp_fft[index]*diff[index]/(np.abs(tmp_fft[index])+self.sigma1)

            self.product[i][self.slice_num-1] = ifftn(tmp_fft) * np.sqrt(np.size(tmp_fft))
            #print 'product [', i, '] and [', self.slice_num-1, '] updated'
            #print np.max(self.product[i][self.slice_num-1] - self.product_old[i][self.slice_num-1])

    # difference map with mode calculation for ptychography
    def recon_dm_trans_mode(self):
        results = self._run_parallel_over_points(self.recon_dm_trans_mode_single)

        for product, result in zip(self.product, results):
            rindex = 0
            for j in range(self.prb_mode_num):
                for k in range(self.obj_mode_num):
                    product[j][k] += result[rindex]
                    rindex += 1

    @profile
    def recon_dm_trans_mode_single(self, i):
        x_start, x_end, y_start, y_end = self.point_info[i]
        diff = self.diff_array[i]

        result = []
        prod_mode = np.zeros((self.nx_prb, self.ny_prb, self.prb_mode_num, self.obj_mode_num)).astype(complex)
        tmp_fft_mode = np.zeros((self.nx_prb, self.ny_prb, self.prb_mode_num, self.obj_mode_num)).astype(complex)

        tmp_fft_mode_total = np.zeros((self.nx_prb, self.ny_prb))
        for j in range(self.prb_mode_num):
            for k in range(self.obj_mode_num):
                prod_mode[:,:, j, k] = self.prb_mode[:,:, j] * self.obj_mode[x_start:x_end, y_start:y_end, k]
                tmp = 2. * prod_mode[:,:, j, k] - self.product[i][j][k]

                tmp_fft_mode[:,:, j, k] = fftn(tmp) / np.sqrt(np.size(tmp))

                tmp_fft_mode_total += np.abs(tmp_fft_mode[:,:, j, k])**2

        for j in range(self.prb_mode_num):
            for k in range(self.obj_mode_num):
                amp_tmp = np.abs(tmp_fft_mode[:,:, j, k])
                ph_tmp = tmp_fft_mode[:,:, j, k] / (amp_tmp+self.sigma1)
                (index_x, index_y) = np.where(diff >= 0.)
                dev = amp_tmp - diff * amp_tmp / (np.sqrt(tmp_fft_mode_total)+self.sigma1)
                power = np.sum((dev[index_x, index_y])**2)/(self.nx_prb*self.ny_prb)
                if (np.sum(diff) > 0.):
                    if power > self.sigma2:
                        a = diff * amp_tmp / (np.sqrt(tmp_fft_mode_total)+self.sigma1) + dev * np.sqrt(self.sigma2/power)
                        amp_tmp[index_x, index_y] = a[index_x, index_y]

                tmp2 = ifftn(amp_tmp*ph_tmp) *  np.sqrt(np.size(tmp))

                result.append(self.beta*(tmp2 - prod_mode[:,:, j, k]))

        return result

    # difference map with mode calculation for ptychography
    def recon_dm_trans_mode_real(self):
        prod_mode = np.zeros((self.nx_prb, self.ny_prb, self.prb_mode_num, self.obj_mode_num)).astype(complex)
        tmp_fft_mode = np.zeros((self.nx_prb, self.ny_prb, self.prb_mode_num, self.obj_mode_num)).astype(complex)

        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            tmp_fft_mode_total = np.zeros((self.nx_prb, self.ny_prb))
            for j in range(self.prb_mode_num):
                for k in range(self.obj_mode_num):
                    prod_mode[:,:, j, k] = self.prb_mode[:,:, j] * self.obj_mode[x_start:x_end, y_start:y_end, k]
                    tmp = 2. * prod_mode[:,:, j, k] - self.product[i][j][k]

                    tmp_fft_mode[:,:, j, k] = fftn(tmp) / np.sqrt(np.size(tmp))

                    tmp_fft_mode_total += np.abs(tmp_fft_mode[:,:, j, k])**2

            for j in range(self.prb_mode_num):
                for k in range(self.obj_mode_num):
                    amp_tmp = np.abs(tmp_fft_mode[:,:, j, k])
                    ph_tmp = tmp_fft_mode[:,:, j, k] / (np.sqrt(tmp_fft_mode_total)+self.sigma1)
                    (index_x, index_y) = np.where(self.diff_array[i] > 0.)
                    amp_tmp[index_x, index_y] = self.diff_array[i][index_x, index_y]

                    tmp2 = ifftn(amp_tmp*ph_tmp) *  np.sqrt(np.size(tmp))

                    self.product[i][j][k] += self.beta*(tmp2 - prod_mode[:,:, j, k])

    # update object with mode
    def cal_object_trans_mode(self):

        for k in range(self.obj_mode_num):
            obj_update = np.zeros([self.nx_obj, self.ny_obj]).astype(complex)
            norm_probe_array = np.zeros((self.nx_obj, self.ny_obj)) + self.alpha

            for j in range(self.prb_mode_num):
                prb_sqr = np.abs(self.prb_mode[:,:, j]) ** 2
                prb_conj = self.prb_mode[:,:, j].conjugate()
                for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                    norm_probe_array[x_start:x_end, y_start:y_end] += prb_sqr
                    obj_update[x_start:x_end, y_start:y_end] += prb_conj * self.product[i][j][k]

            obj_update /= norm_probe_array

            (index_x, index_y) = np.where(abs(obj_update) > self.amp_max_mode[k])
            if(np.size(index_x) > 0):
                obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_max_mode[k] / np.abs(obj_update[index_x, index_y])
            (index_x, index_y) = np.where(abs(obj_update) < self.amp_min_mode[k])
            if(np.size(index_x) > 0):
                obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_min_mode[k] / np.abs(obj_update[index_x, index_y]+1.e-8)

            (index_x, index_y) = np.where(np.angle(obj_update) > self.pha_max_mode[k])
            if(np.size(index_x) > 0):
                obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_max_mode[k])
            (index_x, index_y) = np.where(np.angle(obj_update) < self.pha_min_mode[k])
            if(np.size(index_x) > 0):
                obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_min_mode[k])

            self.obj_mode[:,:, k] = obj_update.copy()

    # update object
    def cal_object_trans(self):

        obj_update = np.zeros((self.nx_obj, self.ny_obj)).astype(complex)
        norm_probe_array = np.zeros((self.nx_obj, self.ny_obj)) + self.alpha

        prb_sqr = np.abs(self.prb) ** 2
        prb_conj = self.prb.conjugate()
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            norm_probe_array[x_start:x_end, y_start:y_end] += prb_sqr
            obj_update[x_start:x_end, y_start:y_end] += prb_conj * self.product[i]

        obj_update /= norm_probe_array
        (index_x, index_y) = np.where(abs(obj_update) > self.amp_max)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_max / np.abs(obj_update[index_x, index_y])
        (index_x, index_y) = np.where(abs(obj_update) < self.amp_min)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_min / np.abs(obj_update[index_x, index_y]+1.e-8)

        (index_x, index_y) = np.where(np.angle(obj_update) > self.pha_max)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_max)
        (index_x, index_y) = np.where(np.angle(obj_update) < self.pha_min)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_min)

        self.obj = obj_update.copy()


    def cal_object_trans_gpu(self):

        points=self.num_points

        for i in range(points) 
            self.kernel_obj_trans( self.product_d, self.obj_d,  self.prb_d,  self.prb_norm_d , \
                self.point_info_d , np.float64(self.alpha) ,  np.int32(self.ny_obj) , np.int32(i), \
                block=(self.ny_prb,1,1), grid=(self.nx_prb,1,1) )

        obj_update =self.obj_d.get()
        norm_probe_array=self.prb_norm_d.get()

        obj_update /= norm_probe_array
        (index_x, index_y) = np.where(abs(obj_update) > self.amp_max)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_max / np.abs(obj_update[index_x, index_y])
        (index_x, index_y) = np.where(abs(obj_update) < self.amp_min)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = obj_update[index_x, index_y] * self.amp_min / np.abs(obj_update[index_x, index_y]+1.e-8)

        (index_x, index_y) = np.where(np.angle(obj_update) > self.pha_max)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_max)
        (index_x, index_y) = np.where(np.angle(obj_update) < self.pha_min)
        if(np.size(index_x) > 0):
            obj_update[index_x, index_y] = np.abs(obj_update[index_x, index_y]) * np.exp(1.j*self.pha_min)

        self.obj = obj_update.copy()

        self.obj_d.set(self.obj)  

    def cal_mass_center(self, array):
        nx, ny = np.shape(array)
        mass_sum = np.sum(np.abs(array))
        mass_sum_x = 0
        for i in range(nx):
            mass_sum_x = mass_sum_x + i * np.sum(np.abs(array[i,:]))
        mass_center_x = np.int(np.round(mass_sum_x / mass_sum))

        mass_sum_y = 0
        for i in range(ny):
            mass_sum_y = mass_sum_y + i * np.sum(np.abs(array[:, i]))
        mass_center_y = np.int(np.round(mass_sum_y / mass_sum))

        return mass_center_x, mass_center_y

    # keep probe well centered
    def check_probe_center(self):
        if self.mode_flag:
            slice_tmp = self.prb_mode[:,:, 0].copy()
            mass_center_x, mass_center_y = self.cal_mass_center(slice_tmp)
            self.prb_mode[:,:, 0] = np.roll(np.roll(slice_tmp, self.nx_prb//2-mass_center_x, 0), self.ny_prb//2-mass_center_y, 1)
            # for j in range(self.prb_mode_num):
            #    slice_tmp = self.prb_mode[:,:,j].copy()
            #    index_tmp = np.where(np.abs(slice_tmp) == np.max(np.abs(slice_tmp)))
            #    self.prb_mode[:,:,j] = np.roll(np.roll(slice_tmp,self.nx_prb/2-index_tmp[0],0),self.ny_prb/2-index_tmp[1],1)
        elif self.multislice_flag:
            slice_tmp = self.prb_ms[0][self.slice_num-1].copy()
            mass_center_x, mass_center_y = self.cal_mass_center(slice_tmp)
            self.prb_ms[0][self.slice_num-1] = np.roll(np.roll(slice_tmp, self.nx_prb//2-mass_center_x, 0), self.ny_prb//2-mass_center_y, 1)
        else:
            # index_tmp = np.where(np.abs(self.prb) == np.max(np.abs(self.prb)))
            # self.prb = np.roll(np.roll(self.prb,self.nx_prb/2-index_tmp[0],0),self.ny_prb/2-index_tmp[1],1)
            mass_center_x, mass_center_y = self.cal_mass_center(self.prb)
            self.prb = np.roll(np.roll(self.prb, np.int(self.nx_prb//2-mass_center_x), 0), np.int(self.ny_prb//2-mass_center_y), 1)

    # update probe with mode
    def cal_probe_trans_mode(self):
        weight = 0.1
        for j in range(self.prb_mode_num):
            probe_update = weight * self.num_points * self.prb_mode[:,:, j]
            norm_obj_array = np.zeros((self.nx_prb, self.ny_prb)) + weight * self.num_points
            for k in range(self.obj_mode_num):
                obj_sqr = np.abs(self.obj_mode[:,:, k]) ** 2
                obj_conj = np.conjugate(self.obj_mode[:,:, k])
                for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                    probe_update += self.product[i][j][k] * obj_conj[x_start:x_end, y_start:y_end]
                    norm_obj_array += obj_sqr[x_start:x_end, y_start:y_end]

            probe_update = probe_update / norm_obj_array
            self.prb_mode[:,:, j] = probe_update.copy()

        if(self.prb_center_flag):
            self.check_probe_center()

    # update probe
    def cal_probe_trans(self):

        weight = 0.1
        probe_update = np.zeros_like(self.prb) #+ weight * self.num_points * self.prb
        norm_obj_array = np.zeros((self.nx_prb, self.ny_prb)) # + weight * self.num_points
        obj_sqr = np.abs(self.obj) ** 2
        obj_conj = np.conjugate(self.obj)
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            probe_update += self.product[i] * obj_conj[x_start:x_end, y_start:y_end]
            norm_obj_array += obj_sqr[x_start:x_end, y_start:y_end]

        probe_update = probe_update / norm_obj_array

        self.prb = probe_update.copy()
        if(self.prb_center_flag):
            self.check_probe_center()

        if self.mask_prb_flag:
            dummy = self.dist_n(self.nx_prb)
            index = np.where(dummy >= (self.nx_prb/2))
            tmp = self.prb[100,100] / 100.
            #print('mask probe', tmp)
            self.prb[index] = tmp

    def ePIE_update(self, f, g, phi, phi_old):
        alpha = 1.
        return (f + alpha * g.conjugate() * (phi - phi_old) / np.max(np.abs(g)**2))

    def multislice_propagate_forward(self):
        print('forward propagation ...')
        for j in range(self.slice_num):
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                if j > 0:
                    self.prb_ms[i][j] = pc.propagate(self.product[i][j-1],1,self.x_pixel_m,self.y_pixel_m,self.lambda_nm*1.e-9,self.slice_spacing_m)
                self.product[i][j] = self.prb_ms[i][j] * self.obj_ms[x_start:x_end, y_start:y_end,j]
            '''
            if j == (self.slice_num-1):
                self.cal_probe_object_trans(j,'object')
                self.cal_probe_object_trans(j,'probe')
            '''

    def cal_probe_trans_gpu_single(self, start_point, size):
        
        #call gpu kernel for calculate prob_update and norm_obj partial sum over points.
        # using memory space allocated as self.obj_prb_d for prob_update
        # and use self.dev_d as space for norm_obj 
        # write accumulated sum to  self.prb_upd_d, self.obj_norm_d,
        tile_pb = 16
        tile_pt = 16
        prb_size = self.nx_prb *self.ny_prb
        offset = np.int32(start_point)

        grid_pb = (prb_size-1)/tile_pb +1
        grid_pt = (size-1)/tile_pt +1 

        #block_size = tile_pb * tile_pt
        #nblocks = grid_pb * grid_pt  
        nx = np.int32(self.nx_prb)
        ny = np.int32(self.ny_prb)
        o_ny = np.int32(self.ny_obj)
        self.kernel_prob_trans( self.product_d,  self.obj_d, \
                self.prb_obj_d, self.dev_d, \
                self.prb_upd_d, self.obj_norm_d, \
                self.point_info_d, \
                np.int32(size), nx, ny, o_ny , offset, \
                block=(tile_pb, tile_pt, 1 ) , grid=(grid_pb, grid_pt, 1)  )
        #self.kernel_prob_trans( self.product_d,  self.obj_d,  self.prb_obj_d, self.dev_d, self.prb_upd_d, self.obj_norm_d, self.point_info_d,  np.int32(size), nx, ny, o_ny , offset,  block=(16,16, 1 ) , grid=(grid_pb, grid_pt, 1)  )
        #self.kernel_prob_trans( self.product_d,  self.obj_d,  self.prb_obj_d, self.dev_d, self.point_info_d,  np.int32(size), nx, ny, o_ny , offset,  block=(16,16, 1 ) , grid=(grid_pb, grid_pt, 1)  )
        
        #further reduce :
        #block_size = tile_pb * grid_pt
        if  grid_pt > 1  :
            self.kernel_prob_reduce(self.prb_obj_d, self.dev_d , self.prb_upd_d, self.obj_norm_d, np.int32(grid_pt) , np.int32(prb_size), offset,\
                block=(tile_pb, grid_pt, 1) , grid=( grid_pb, 1,1 ) )

    def cal_probe_trans_gpu(self) :


        #load obj to GPU , only need before obj update is done in GPU
        self.obj_d.set(self.obj) 

        n_batch = self.num_points/self.gpu_batch_size 
        
        for i in range(n_batch) :
            self.cal_probe_trans_gpu_single( i*self.gpu_batch_size, self.gpu_batch_size)

        if self.last >0 :
            self.cal_probe_trans_gpu_single( n_batch*self.gpu_batch_size, self.last )


        # read the prob_update and normal_obj out
        norm=self.obj_norm_d.get()
        prb=self.prb_upd_d.get() 
        #update prb
        self.prb = prb/norm
        # load back prb to GPU , this is necessary for multi GPU implementation later on
        self.prb_d.set(self.prb)
        # 



    def multislice_propagate_backward(self,it):
        print('backward propagation ...')
        for j in range(self.slice_num):
            if j > 0:
                for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                    self.product[i][self.slice_num-1-j] = \
                        pc.propagate(self.prb_ms[i][self.slice_num-j],\
                                     -1,self.x_pixel_m,self.y_pixel_m,self.lambda_nm*1.e-9,self.slice_spacing_m)
            #self.cal_probe_object_trans(self.slice_num-1-j,'object')
            #self.cal_probe_object_trans(self.slice_num-1-j,'probe')
            if(it >= self.start_update_probe):
                if(it >= self.start_update_object):
                    self.cal_probe_object_trans(self.slice_num-1-j,'object')
                    self.cal_probe_object_trans(self.slice_num-1-j,'probe')
                else:
                    self.cal_probe_object_trans(self.slice_num-1-j,'probe')
            else:
                if(it >= self.start_update_object):
                    self.cal_probe_object_trans(self.slice_num-1-j,'object')
                if ((self.slice_num-1-j) != 0):
                    self.cal_probe_object_trans(self.slice_num-1-j,'probe')

    # update object or probe for multislice
    def cal_probe_object_trans(self, slice_num, sign):
        weight = 0.1
        if sign == 'probe':
            array_sqr = np.abs(self.obj_ms[:,:,slice_num]) ** 2
            array_conj = np.conjugate(self.obj_ms[:,:,slice_num])
            if slice_num == 0:
                array_update = weight * self.num_points * self.prb_ms[0][slice_num]
                norm_array = np.zeros((self.nx_prb, self.ny_prb)) + weight * self.num_points
        else:
            array_update = np.zeros((self.nx_obj, self.ny_obj)).astype(complex)
            norm_array = np.zeros((self.nx_obj, self.ny_obj)) + self.alpha

        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            if sign == 'object':
                array_sqr = np.abs(self.prb_ms[i][slice_num]) ** 2
                array_conj = np.conjugate(self.prb_ms[i][slice_num])
                norm_array[x_start:x_end, y_start:y_end] += array_sqr
                array_update[x_start:x_end, y_start:y_end] += array_conj * self.product[i][slice_num]
            else:
                if slice_num == 0:
                    array_update += array_conj[x_start:x_end, y_start:y_end] * self.product[i][slice_num]
                    norm_array += array_sqr[x_start:x_end, y_start:y_end]
                else:
                    #norm_array = np.zeros((self.nx_prb, self.ny_prb)) + 1.e-9
                    #array_update = self.prb_ms[i][slice_num]
                    #norm_array += array_sqr[x_start:x_end, y_start:y_end]
                    #array_update += array_conj[x_start:x_end, y_start:y_end] * self.product[i][slice_num]
                    norm_array = array_sqr[x_start:x_end, y_start:y_end] + 1.e-9
                    array_update = array_conj[x_start:x_end, y_start:y_end] \
                            * self.product[i][slice_num]
                    self.prb_ms[i][slice_num] = array_update.copy() / norm_array

        if (slice_num == 0) and (sign == 'probe'):
            array_update = array_update / norm_array
            for i in range(self.num_points):
                self.prb_ms[i][slice_num] = array_update.copy()

        if sign == 'object':
            array_update = array_update / norm_array
            (index_x, index_y) = np.where(abs(array_update) > self.amp_max_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = array_update[index_x, index_y] * self.amp_max_ms[slice_num] / np.abs(array_update[index_x, index_y])
            (index_x, index_y) = np.where(abs(array_update) < self.amp_min_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = array_update[index_x, index_y] * self.amp_min_ms[slice_num] / np.abs(array_update[index_x, index_y]+1.e-8)

            (index_x, index_y) = np.where(np.angle(array_update) > self.pha_max_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = np.abs(array_update[index_x, index_y]) * np.exp(1.j*self.pha_max_ms[slice_num])
            (index_x, index_y) = np.where(np.angle(array_update) < self.pha_min_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = np.abs(array_update[index_x, index_y]) * np.exp(1.j*self.pha_min_ms[slice_num])

            if self.current_it <= 20:
                mask_1 = np.load('21689_mask_1.npy')
                mask_2 = np.load('21689_mask_2.npy')
                mask_bg = np.load('21689_mask_bg.npy')
                bg = np.mean(array_update[mask_bg == 1.])
                if slice_num == 1:
                    array_update[mask_1 == 1.] = bg
                else:
                    array_update[mask_2 == 1.] = bg


            self.obj_ms[:,:,slice_num] = array_update.copy()

    def cal_probe_object_trans_bp(self, slice_num, sign):

        if sign == 'probe':
            array_sqr = np.abs(self.obj_ms[:,:,slice_num]) ** 2
            array_conj = (self.obj_ms[:,:,slice_num]).conjugate()
        else:
            array_update = np.zeros((self.nx_obj, self.ny_obj)).astype(complex)
            norm_array = np.zeros((self.nx_obj, self.ny_obj)) + self.alpha

        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            if sign == 'object':
                array_sqr = np.abs(self.prb_ms[i][slice_num]) ** 2
                array_conj = (self.prb_ms[i][slice_num]).conjugate()
                norm_array[x_start:x_end, y_start:y_end] += array_sqr
                array_update[x_start:x_end, y_start:y_end] += array_conj * self.product[i][slice_num]
            else:
                norm_array = array_sqr[x_start:x_end, y_start:y_end]
                array_update = array_conj[x_start:x_end, y_start:y_end] * self.product[i][slice_num]
                self.prb_ms[i][slice_num] = array_update.copy() / (norm_array+ self.alpha)

        if (slice_num == 0) and (sign == 'probe'):
            tmp = np.zeros((self.nx_prb,self.ny_prb)).astype(complex)
            for ii in range(self.nx_prb):
                for jj in range(self.ny_prb):
                    tmp_tmp = np.complex(0,0)
                    for kk in range(self.num_points):
                        tmp_tmp += self.prb_ms[kk][slice_num][ii,jj]
                    tmp[ii,jj] = tmp_tmp/self.num_points
            for i in range(self.num_points):
                self.prb_ms[i][slice_num] = tmp.copy()

        if sign == 'object':
            array_update /= norm_array
            (index_x, index_y) = np.where(abs(array_update) > self.amp_max_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = array_update[index_x, index_y] * self.amp_max_ms[slice_num] / np.abs(array_update[index_x, index_y])
            (index_x, index_y) = np.where(abs(array_update) < self.amp_min_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = array_update[index_x, index_y] * self.amp_min_ms[slice_num] / np.abs(array_update[index_x, index_y]+1.e-8)

            (index_x, index_y) = np.where(np.angle(array_update) > self.pha_max_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = np.abs(array_update[index_x, index_y]) * np.exp(1.j*self.pha_max_ms[slice_num])
            (index_x, index_y) = np.where(np.angle(array_update) < self.pha_min_ms[slice_num])
            if(np.size(index_x) > 0):
                array_update[index_x, index_y] = np.abs(array_update[index_x, index_y]) * np.exp(1.j*self.pha_min_ms[slice_num])

            self.obj_ms[:,:,slice_num] = array_update.copy()


    def cal_position_correction(self):
        correct_points = np.load('./points_fermat_n.npy') + 20
        tmp = 0.
        for i in range(self.num_points):
            tmp = tmp + np.sqrt((correct_points[0, i]-self.points[0, i])**2+(correct_points[1, i]-self.points[1, i])**2)

        print('points error:', tmp)

    # update position
    def position_correction(self):

        count = 0
        points_new = self.points.copy()
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            error_ref = 1.e8
            # norm_diff = np.sqrt(np.sum(np.abs(self.diff_array[i])**2)/(self.nx_prb*self.ny_prb))
            for ix in range(-1*self.position_correction_search_range, self.position_correction_search_range+1):
                for iy in range(-1*self.position_correction_search_range, self.position_correction_search_range+1):
                    prod_tmp = np.abs(fftn(self.prb * self.obj[x_start+ix:x_end+ix, y_start+iy:y_end+iy]) / np.sqrt(1.*np.size(self.prb)))

                    # norm_tmp = np.sqrt(np.sum(np.abs(prod_tmp)**2)/(self.nx_prb*self.ny_prb))
                    error_tmp = np.sqrt(np.sum(np.abs(prod_tmp - self.diff_array[i])**2)/np.sum(np.abs(self.diff_array[i])**2))
                    if(error_tmp < error_ref):
                        error_ref = error_tmp.copy()
                        if(x_start+ix >= self.position_correction_search_range and x_end+ix <= self.nx_obj-self.position_correction_search_range):
                            points_new[0, i] = self.points[0, i] + ix
                        if(y_start+iy >= self.position_correction_search_range and y_end+iy <= self.ny_obj-self.position_correction_search_range):
                            points_new[1, i] = self.points[1, i] + iy
                    # print error_tmp, error_ref,points_new[:,i]

            # print self.points[:,i],points_new[:,i]
            if((points_new[0, i] != self.points[0, i]) or (points_new[1, i] != self.points[1, i])):
                count += 1

        list_tmp = self.points_list[0, 0,:]
        index_tmp = np.where(list_tmp == 0.)
        self.points_list[:,:, index_tmp[0][0]] = points_new.copy()
        self.points = points_new.copy()
        self.point_info = np.array([(int(self.points[0, i] - self.nx_prb//2), int(self.points[0, i] + self.nx_prb//2), \
                                      int(self.points[1, i] - self.ny_prb//2), int(self.points[1, i] + self.ny_prb//2)) \
            for i in range(self.num_points)])

        return count

    # update position for mode
    def position_correction_mode(self):

        count = 0
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            error_ref = 1.e8
            ix_tmp = 0
            iy_tmp = 0
            norm_diff = np.sqrt(np.sum(np.abs(self.diff_array[i])**2)/(self.nx_prb*self.ny_prb))
            for ix in range(-1*self.position_correction_search_range, self.position_correction_search_range+1):
                for iy in range(-1*self.position_correction_search_range, self.position_correction_search_range+1):
                    prod_tmp = np.zeros((self.nx_prb, self.ny_prb))
                    for j in range(self.prb_mode_num):
                        for k in range(self.obj_mode_num):
                            prod_tmp += np.abs(fftn(np.reshape(self.prb_mode[:,:, j], (self.nx_prb, self.ny_prb)) * np.reshape(self.obj_mode[x_start+ix:x_end+ix, y_start+iy:y_end+iy, k], (self.nx_prb, self.ny_prb))) / np.sqrt(np.size(self.prb)))**2

                    norm_tmp = np.sqrt(np.sum(np.abs(prod_tmp)**2)/(self.nx_prb*self.ny_prb))
                    prod_tmp = np.sqrt(prod_tmp / norm_tmp) * norm_diff
                    error_tmp = np.sqrt(np.sum(np.abs(prod_tmp - self.diff_array[i])**2)/np.sum(np.abs(self.diff_array[i])**2))

                    if(error_tmp < error_ref):
                        error_ref = error_tmp.copy()
                        if(x_start+ix >= self.position_correction_search_range and x_end+ix <= self.nx_obj-self.position_correction_search_range):
                            ix_tmp = ix
                        if(y_start+iy >= self.position_correction_search_range and y_end+iy <= self.ny_obj-self.position_correction_search_range):
                            iy_tmp = iy

            if(ix_tmp != 0 or iy_tmp != 0):
                self.points[0, i] = self.points[0, i] + ix_tmp
                self.points[1, i] = self.points[1, i] + iy_tmp
                count += 1

        self.point_info = np.array([(int(self.points[0, i] - self.nx_prb//2), int(self.points[0, i] + self.nx_prb//2), \
                                      int(self.points[1, i] - self.ny_prb//2), int(self.points[1, i] + self.ny_prb//2)) \
            for i in range(self.num_points)])

        return count

    def cal_obj_prb_dim(self):

        if self.mode_flag:
            self.error_prb_mode = np.zeros((self.n_iterations, self.prb_mode_num))
            self.error_obj_mode = np.zeros((self.n_iterations, self.obj_mode_num))
        if self.multislice_flag:
            self.error_prb_ms = np.zeros((self.n_iterations))
            self.error_obj_ms = np.zeros((self.n_iterations, self.slice_num))
        else:
            self.error_obj = np.zeros(self.n_iterations)
            self.error_prb = np.zeros(self.n_iterations)
            if self.pc_flag:
                self.error_coh = np.zeros(self.n_iterations)
        self.error_chi = np.zeros(self.n_iterations)
        if self.position_correction_flag:
            self.points_ini = self.points.copy()
            self.points_list = np.zeros((2, self.num_points, self.n_iterations/self.position_correction_step))

        self.nx_obj = self.x_roi + np.max(self.points[0,:]) - np.min(self.points[0,:]) + self.obj_pad
        self.ny_obj = self.y_roi + np.max(self.points[1,:]) - np.min(self.points[1,:]) + self.obj_pad

        print(self.nx_obj, self.ny_obj)
        self.nx_obj = np.int(self.nx_obj + np.mod(self.nx_obj, 2))
        self.ny_obj = np.int(self.ny_obj + np.mod(self.ny_obj, 2))
        if self.cal_scan_pattern_flag:
            self.points[0,:] = self.points[0,:] + self.nx_obj // 2
            self.points[1,:] = self.points[1,:] + self.ny_obj // 2

        self.point_info = np.array([(int(self.points[0, i] - self.nx_prb//2), int(self.points[0, i] + self.nx_prb//2), \
                                      int(self.points[1, i] - self.ny_prb//2), int(self.points[1, i] + self.ny_prb//2)) \
                                for i in range(self.num_points)])

    def init_obj(self):
        if self.mode_flag:
            self.obj_mode = np.zeros((self.nx_obj, self.ny_obj, self.obj_mode_num)).astype(complex)
            for i in range(self.obj_mode_num):
                self.obj_mode[:,:, i] = np.random.uniform(0, 0.5, (self.nx_obj, self.ny_obj)) * \
                    np.exp(np.random.uniform(0, 0.5, (self.nx_obj, self.ny_obj))*1.j)
        elif self.multislice_flag:
            self.obj_ms = np.zeros((self.nx_obj, self.ny_obj, self.slice_num)).astype(complex)
            for i in range(self.slice_num):
                self.obj_ms[:,:, i] = np.random.uniform(0, 1., (self.nx_obj, self.ny_obj)) * \
                    np.exp(np.random.uniform(0, 1., (self.nx_obj, self.ny_obj))*1.j)
        else:
            print('init object')
            self.obj = np.random.uniform(0, 1, (self.nx_obj, self.ny_obj)) * \
                np.exp(np.random.uniform(0, np.pi, (self.nx_obj, self.ny_obj))*1.j)
            #self.obj[self.nx_obj/2-80:self.nx_obj/2+80,self.ny_obj/2-80:self.ny_obj/2+80] = 1.

    def init_obj_stxm_dpc(self):
        if self.mode_flag:
            self.obj_mode = np.zeros((self.nx_obj, self.ny_obj, self.obj_mode_num)).astype(complex)
            for i in range(self.obj_mode_num):
                self.obj_mode[:,:, i] = np.random.uniform(0, 0.5, (self.nx_obj, self.ny_obj)) * \
                    np.exp(np.random.uniform(0, 0.5, (self.nx_obj, self.ny_obj))*1.j)
        elif self.multislice_flag:
            self.obj_ms = np.zeros((self.nx_obj, self.ny_obj, self.slice_num)).astype(complex)
            for i in range(self.slice_num):
                self.obj_ms[:,:, i] = np.random.uniform(0, 1., (self.nx_obj, self.ny_obj)) * \
                    np.exp(np.random.uniform(0, 1., (self.nx_obj, self.ny_obj))*1.j)
        else:
            print('init object with stxm and dpc')
            diff_array = np.load('diff_data_'+self.scan_num+'.npy')
            obj_core,tt = cal_stxm_dpc(diff_array, self.ccd_pixel_um, self.z_m, self.lambda_nm, \
                    self.dpc_x_step_m, self.dpc_y_step_m, self.dpc_col, self.dpc_row, \
                    crop_size=self.dpc_crop_size,x_flip=self.dpc_x_flip,y_flip=self.dpc_y_flip)
            print(self.dpc_col,self.dpc_row,self.dpc_x_step_m,self.dpc_y_step_m,self.x_pixel_m)
            nx_new = np.round(self.dpc_col * self.dpc_x_step_m / self.x_pixel_m)
            nx_new += np.mod(nx_new, 2)
            ny_new = np.round(self.dpc_row * self.dpc_x_step_m / self.y_pixel_m)
            ny_new += np.mod(ny_new, 2)
            obj_core_new = self.congrid_fft(obj_core,(nx_new,ny_new))
            self.obj = np.random.uniform(0, 1, (self.nx_obj, self.ny_obj)) * \
                np.exp(np.random.uniform(0, np.pi, (self.nx_obj, self.ny_obj))*1.j)
            print(np.shape(obj_core_new),np.shape(self.obj))
            self.obj[self.nx_obj//2-nx_new//2:self.nx_obj//2+nx_new//2,self.ny_obj//2-ny_new//2:self.ny_obj//2+ny_new//2] = \
                obj_core_new
            np.save('obj_init_dpc.npy',self.obj)

    def init_prb(self):
        print('initialize probe')
        if self.mode_flag:
            self.prb_mode = np.zeros((self.nx_prb, self.ny_prb, self.prb_mode_num)).astype(complex)
            for i in range(self.prb_mode_num):
                self.prb_mode[:,:, i] = np.random.uniform(0, 0.5, (self.nx_prb, self.ny_prb)) * \
                    np.exp(np.random.uniform(0, 0.5, (self.nx_prb, self.ny_prb))*1.j)
                self.prb_mode[self.nx_prb//2-10:self.nx_prb//2+10, self.ny_prb//2-10:self.ny_prb//2+10, i] = 1.
        elif self.multislice_flag:
            #dummy = self.dist_n(self.nx_prb)
            #index = np.where(dummy <= 35.)
            self.prb_ms = [0 for i in range(self.num_points)]
            self.prb_ms_old = [0 for i in range(self.num_points)]
            self.prb_ms_ave = [0 for i in range(self.num_points)]
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                self.prb_ms[i] = [0 for ii in range(self.slice_num)]
                self.prb_ms_old[i] = [0 for ii in range(self.slice_num)]
                self.prb_ms_ave[i] = [0 for ii in range(self.slice_num)]
                for j in range(self.slice_num):
                    self.prb_ms[i][j] = np.zeros((self.nx_prb, self.ny_prb)).astype(complex) + 100.
                    self.prb_ms_old[i][j] = np.zeros((self.nx_prb, self.ny_prb)).astype(complex)
                    self.prb_ms_ave[i][j] = np.zeros((self.nx_prb, self.ny_prb)).astype(complex)
                    if j == 0:
                        #self.prb_ms[i][j][self.nx_prb/2-20:self.nx_prb/2+20, self.ny_prb/2-10:self.ny_prb/2+10] = 100.
                        #self.prb_ms[i][j][index] = 1.e5 #100000.
                        self.prb_ms[i][j] = self.prb_ini  #np.load('MLL_probe_128_n.npy')
        else:
            self.prb = np.random.uniform(0, 1, (self.nx_prb, self.ny_prb)) * \
                    np.exp(np.random.uniform(0, 3, (self.nx_prb, self.ny_prb))*1.j)
            #self.prb = np.zeros((self.nx_prb,self.ny_prb)).astype(complex) + 0.5
            dummy = self.dist_n(self.nx_prb)
            index = np.where(dummy <= 2.)
            self.prb[self.nx_prb//2-25:self.nx_prb//2+25, self.ny_prb//2-25:self.ny_prb//2+25] = 100.
            #self.prb[index] = 100.
            #core = self.create_gauss_2d(40) * np.exp(1j)
            #self.prb = core * 10000.

    def init_product(self):
        self.position_correction_flag_ini = self.position_correction_flag
        if self.mode_flag:
            self.amp_max_mode = np.zeros(self.obj_mode_num)
            self.amp_min_mode = np.zeros(self.obj_mode_num)
            self.pha_max_mode = np.zeros(self.obj_mode_num)
            self.pha_min_mode = np.zeros(self.obj_mode_num)
            for i in range(self.obj_mode_num):
                self.amp_max_mode[i] = self.amp_max
                self.amp_min_mode[i] = self.amp_min
                self.pha_max_mode[i] = self.pha_max
                self.pha_min_mode[i] = self.pha_min
            self.product = [0 for i in range(self.num_points)]
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                self.product[i] = [0 for ii in range(self.prb_mode_num)]
                for j in range(self.prb_mode_num):
                    self.product[i][j] = [0 for ii in range(self.obj_mode_num)]
                    for k in range(self.obj_mode_num):
                        self.product[i][j][k] = self.prb_mode[:,:, j] * self.obj_mode[x_start:x_end, y_start:y_end, k]
        elif self.multislice_flag:
            self.amp_max_ms = self.amp_max
            self.amp_min_ms = self.amp_min
            self.pha_max_ms = self.pha_max
            self.pha_min_ms = self.pha_min
            '''
            self.amp_max_ms = np.zeros(self.slice_num)
            self.amp_min_ms = np.zeros(self.slice_num)
            self.pha_max_ms = np.zeros(self.slice_num)
            self.pha_min_ms = np.zeros(self.slice_num)
            for i in range(self.slice_num):
                self.amp_max_ms[i] = self.amp_max
                self.amp_min_ms[i] = self.amp_min
                self.pha_max_ms[i] = self.pha_max
                self.pha_min_ms[i] = self.pha_min
            '''
            self.product = [0 for i in range(self.num_points)]
            self.product_old = [0 for i in range(self.num_points)]
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                self.product[i] = [0 for ii in range(self.slice_num)]
                self.product_old[i] = [0 for ii in range(self.slice_num)]
                for j in range(self.slice_num):
                    #print(np.shape(self.prb_ms),np.shape(self.obj_ms))
                    self.product[i][j] = self.prb_ms[i][j] * self.obj_ms[x_start:x_end, y_start:y_end,j]
                    self.product_old[i][j] = self.product[i][j].copy()
        else:
            self.product = [0 for i in range(self.num_points)]
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                self.product[i] = self.prb * self.obj[x_start:x_end, y_start:y_end]

    def init_coh(self):
        self.coh = self.create_gauss((self.pc_kernel_n,self.pc_kernel_n),self.pc_sigma)
        self.coh /= np.sum(self.coh)

    # initiate filter function
    def init_pc_filter(self):
        #self.pc_filter = self.create_gauss((self.coh_n,self.coh_n),self.pc_filter_sigma)
        #self.pc_filter /= np.max(self.pc_filter)
        self.pc_filter = self.create_tukey((self.pc_kernel_n,self.pc_kernel_n))

    def cal_obj_error(self, it):
        if self.mode_flag:
            for j in range(self.obj_mode_num):
                self.error_obj_mode[it, j] = np.sqrt(np.sum(np.abs(self.obj_mode[:,:, j] - self.obj_mode_old[:,:, j])**2)) / \
                    np.sqrt(np.sum(np.abs(self.obj_mode[:,:, j])**2))
        elif self.multislice_flag:
            for j in range(self.slice_num):
                self.error_obj_ms[it, j] = np.sqrt(np.sum(np.abs(self.obj_ms[:,:, j] - self.obj_ms_old[:,:, j])**2)) / \
                    np.sqrt(np.sum(np.abs(self.obj_ms[:,:, j])**2))
        else:
            self.error_obj[it] = np.sqrt(np.sum(np.abs(self.obj - self.obj_old)**2)) / \
                np.sqrt(np.sum(np.abs(self.obj)**2))

    def cal_prb_error(self, it):
        if self.mode_flag:
            for j in range(self.prb_mode_num):
                self.error_prb_mode[it, j] = np.sqrt(np.sum(np.abs(self.prb_mode[:,:, j] - self.prb_mode_old[:,:, j])**2)) / \
                    np.sqrt(np.sum(np.abs(self.prb_mode[:,:, j])**2))
        elif self.multislice_flag:
            '''
            for j in range(self.slice_num):
                error_tmp = 0.
                for i in range(self.num_points):
                    error_tmp += np.sqrt(np.sum(np.abs(self.prb_ms[i][j] - self.prb_ms_old[i][j])**2)) / \
                        np.sqrt(np.sum(np.abs(self.prb_ms[i][j])**2))
                self.error_prb_ms[it, j] = error_tmp / self.num_points
            '''
            self.error_prb_ms[it] = np.sqrt(np.sum(np.abs(self.prb_ms[0][0] - self.prb_ms_old[0][0])**2)) / \
                np.sqrt(np.sum(np.abs(self.prb_ms[0][0])**2))
        else:
            self.error_prb[it] = np.sqrt(np.sum(np.abs(self.prb - self.prb_old)**2)) / \
                np.sqrt(np.sum(np.abs(self.prb)**2))

    def cal_chi_error(self, it):
        chi_tmp = 0.
        if self.mode_flag:
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                tmp = np.zeros((self.nx_prb, self.ny_prb))
                for j in range(self.prb_mode_num):
                    for k in range(self.obj_mode_num):
                        tmp = tmp + (np.abs(fftn(self.prb_mode[:,:, j]*self.obj_mode[x_start:x_end, y_start:y_end, k])/np.sqrt(1.*self.nx_prb*self.ny_prb)))**2
                if np.sum((self.diff_array[i])**2) > 0.:
                    chi_tmp = chi_tmp + np.sum((np.sqrt(tmp) - self.diff_array[i])**2)/(np.sum((self.diff_array[i])**2))
        elif self.multislice_flag:
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                tmp = np.abs(fftn(self.prb_ms[i][self.slice_num-1]*self.obj_ms[x_start:x_end, y_start:y_end,self.slice_num-1])/\
                    np.sqrt(1.*self.nx_prb*self.ny_prb))
                if np.sum((self.diff_array[i])**2) > 0.:
                    chi_tmp = chi_tmp + np.sum((tmp - self.diff_array[i])**2)/(np.sum((self.diff_array[i])**2))
        else:
            for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
                tmp = np.abs(fftn(self.prb*self.obj[x_start:x_end, y_start:y_end])/np.sqrt(1.*self.nx_prb*self.ny_prb))
                if np.sum((self.diff_array[i])**2) > 0.:
                    chi_tmp = chi_tmp + np.sum((tmp - self.diff_array[i])**2)/(np.sum((self.diff_array[i])**2))

        self.error_chi[it] = np.sqrt(chi_tmp/self.num_points)


    def chi_gpu_single(self, start_point, size) :
        
        offset = np.int32(start_point) 
        block_size = self.ny_prb
        n_blocks = size *self.nx_prb
	nx = np.int32(self.nx_prb)
	ny = np.int32(self.ny_prb)
	o_ny = np.int32(self.ny_obj)
        self.kernel_chi_prb_obj(self.prb_d, self.obj_d, self.fft_tmp_d, self.point_info_d, \
		nx, ny, o_ny, offset, \
                block=(block_size,1,1), grid=(n_blocks,1,1) )

        cu_fft.fft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f )

        block_size=self.ny_prb
        nblocks = self.nx_prb  * size 
        chi_tmp=np.zeros((nblocks,),dtype=np.float64) 
        buff =gpuarray.empty(np.shape(chi_tmp),dtype=np.float64)
        scale = np.int32(self.nx_prb*self.ny_prb )
        self.kernel_chi_sum_block(self.fft_tmp_d, self.diff_d, self.diff_sum_sq_d, buff, \
                 scale, offset, block=(block_size,1,1 ) , grid=(nblocks,1,1) , shared=block_size*8  )
        chi_tmp = buff.get() 
        return np.sum(chi_tmp) 

    def cal_chi_error_gpu(self, it) :
        chi=0.0
        n_batch = self.num_points/self.gpu_batch_size
         
        self.prb_d.set(self.prb)
        self.obj_d.set(self.obj) 
        #cuda.memcpy_htod(self.prb_d, self.prb )
        #cuda.memcpy_htod(self.obj_d, self.obj )
        for i in range(n_batch) :
            chi += self.chi_gpu_single( i*self.gpu_batch_size , self.gpu_batch_size ) 
        if  self.last > 0 :
            chi += self.chi_gpu_single( n_batch*self.gpu_batch_size, self.last ) 
        self.error_chi[it] = np.sqrt(chi/self.num_points)
    
        

    def cal_chi_error_gpu_save(self, it):
	t1 = time.time()
        chi = 0.0
        scale_sqrt=np.sqrt(1.*self.nx_prb*self.ny_prb)
        scale = np.int32(self.nx_prb*self.ny_prb )
#        chi_tmp_cpu = np.empty_like(self.product)

        cuda.memcpy_htod(self.prb_d, self.prb )
        cuda.memcpy_htod(self.obj_d, self.obj )


        block_size = self.ny_prb
        n_blocks = self.num_points*self.nx_prb*self.ny_prb/block_size
	
	nx = np.int32(self.nx_prb)
	ny = np.int32(self.ny_prb)
	o_ny = np.int32(self.ny_obj)
	points = np.int32(self.num_points)
	cuda.Context.synchronize()
        t2=time.time()
        self.elaps[9] += t2-t1

        self.kernel_chi_prb_obj(self.prb_d, self.obj_d, self.fft_tmp_d, self.point_info_d, \
		nx, ny, o_ny, points, \
                block=(block_size,1,1), grid=(n_blocks,1,1) )
#        cuda.memcpy_dtoh(chi_tmp, self.fft_tmp_d.gpudata )
	cuda.Context.synchronize()
        t3=time.time()
        self.elaps[10] += t3-t2


        cu_fft.fft(self.fft_tmp_d, self.fft_tmp_d, self.plan_f )
	cuda.Context.synchronize()
        t1=time.time()
        self.elaps[11] += t1-t3
#        chi_tmp_cpu =self.fft_tmp_d.get()

	cuda.Context.synchronize()
        t0 = time.time()

	self.elaps[12] += t0-t1

        # choose a block size for calculate chi sum for each  block.
        # it will end up with nblock numbers 
        block_size=1024
        nblocks = (self.nx_prb * self.ny_prb* self.num_points -1 )/block_size + 1

        chi_tmp=np.zeros((nblocks,),dtype=np.float64) 
        buff =gpuarray.empty(np.shape(chi_tmp),dtype=np.float64)
        self.kernel_chi_sum_block(self.fft_tmp_d, self.diff_d, self.diff_sum_sq_d, buff, \
            scale, block=(block_size,1,1 ) , grid=(nblocks,1,1) , shared=block_size*8  )
	cuda.Context.synchronize()
        t5 = time.time()
        chi_tmp = buff.get() 

        '''
        # Not sure why it took very long time to do gpuarray.sum
        t11=time.time()
        chi_g=gpuarray.sum(buff)
        chi=float(chi_g.get()) 
        print "chi gpu=", chi
        print "time " , time.time()-t11
        '''
        ### number of GOU blocks for futher reduce to nblock_reduce numbers
        nblock_reduce=(nblocks-1)/block_size +1 

                
        '''
        buff = cuda.mem_alloc(nblock_reduce*8  )        
        self.kernel_chi_reduce(self.prb_obj_d, buff, np.int32(nblocks),\
            block=(block_size, 1,1 ) , grid=(nblock_reduce,1,1 ), shared=block_size*8 )

        chi_tmp=np.empty(nblock_reduce,dtype=np.float64) 
        cuda.memcpy_dtoh(chi_tmp, buff) 
        '''
	cuda.Context.synchronize()
        self.elaps[6] += t5-t0
        
        chi = np.sum(chi_tmp) 
        print "chi cpu=", chi
        '''
        for i in range(self.num_points):
            tmp = np.sum((np.abs(chi_tmp[i])/scale_sqrt - self.diff_array[i])**2)
            chi += tmp/self.diff_sum_sq[i]
        '''        
        self.elaps[7] += time.time()-t5 
             
        self.error_chi[it] = np.sqrt(chi/self.num_points)


    def cal_coh_error(self, it):
        self.error_coh[it] = np.sqrt(np.sum(np.abs(self.coh - self.coh_old)**2)) / \
            np.sqrt(np.sum(np.abs(self.coh)**2))

    def save_recon(self):

        save_dir = './recon_result/S'+self.scan_num+'/'+self.sign+'/recon_data/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_pic_dir = './recon_result/S'+self.scan_num+'/'+self.sign+'/recon_pic/'
        if not os.path.exists(save_pic_dir):
            os.makedirs(save_pic_dir)

        if self.mode_flag:
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object_ave', self.obj_mode_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object', self.obj_mode)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_object', self.error_obj_mode)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe_ave', self.prb_mode_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe', self.prb_mode)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_probe', self.error_prb_mode)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_chi', self.error_chi)
            text_file = open(save_dir+'recon_message.txt', 'w')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.write('object size: '+np.str(self.nx_obj)+' x '+np.str(self.ny_obj)+'\n')
            text_file.write('probe size: '+np.str(self.nx_prb)+' x '+np.str(self.ny_prb)+'\n')
            text_file.write('real space pixel size: '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um))+' x '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um))+' nm \n')
            text_file.write('probe/object modes number: '+np.str(self.prb_mode_num)+' / '+np.str(self.obj_mode_num)+' \n')
            text_file.write('total scan points: '+np.str(self.num_points)+'\n')
            text_file.write(np.str(self.n_iterations)+' iterations take '+np.str(self.time_end - self.time_start)+' sec'+'\n')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.close()
        elif self.multislice_flag:
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object_ave', self.obj_ms_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object', self.obj_ms)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_object', self.error_obj_ms)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe_ave', self.prb_ms_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe', self.prb_ms)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_probe', self.error_prb_ms)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_chi', self.error_chi)
            text_file = open(save_dir+'recon_message.txt', 'w')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.write('object size: '+np.str(self.nx_obj)+' x '+np.str(self.ny_obj)+'\n')
            text_file.write('probe size: '+np.str(self.nx_prb)+' x '+np.str(self.ny_prb)+'\n')
            text_file.write('real space pixel size: '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um))+' x '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um))+' nm \n')
            text_file.write('slice number: '+np.str(self.slice_num)+' \n')
            text_file.write('total scan points: '+np.str(self.num_points)+'\n')
            text_file.write(np.str(self.n_iterations)+' iterations take '+np.str(self.time_end - self.time_start)+' sec'+'\n')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.close()
        else:
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object_ave', self.obj_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_object', self.obj)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_object', self.error_obj)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe_ave', self.prb_ave)
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe', self.prb)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_probe', self.error_prb)
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_chi', self.error_chi)
            text_file = open(save_dir+'recon_message.txt', 'w')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.write('object size: '+np.str(self.nx_obj)+' x '+np.str(self.ny_obj)+'\n')
            text_file.write('probe size: '+np.str(self.nx_prb)+' x '+np.str(self.ny_prb)+'\n')
            text_file.write('real space pixel size: '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um))+' x '+np.str(self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um))+' nm \n')
            text_file.write('total scan points: '+np.str(self.num_points)+'\n')
            text_file.write(np.str(self.n_iterations)+' iterations take '+np.str(self.time_end - self.time_start)+' sec'+'\n')
            text_file.write('++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            text_file.close()
        if(self.update_coh_flag):
            np.save(save_dir+'error_'+self.scan_num+'_'+self.sign+'_coh', self.error_coh)
        # sm.imsave(save_pic_dir+'recon_'+self.scan_num+'_'+self.sign+'_object.jpg',abs(self.obj_ave))
        # sm.imsave(save_pic_dir+'recon_'+self.scan_num+'_'+self.sign+'_probe.jpg',abs(self.prb_ave))
        np.save(save_dir+'time_'+self.scan_num+'_'+self.sign, self.time_end-self.time_start)
        if self.pc_flag:
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_coh_fft', self.coh)
            tmp = np.zeros((self.nx_prb, self.ny_prb))
            tmp[self.nx_prb/2-self.pc_kernel_n//2:self.nx_prb//2+self.pc_kernel_n//2, \
                self.ny_prb/2-self.pc_kernel_n//2:self.ny_prb//2+self.pc_kernel_n//2] = self.coh
            self.coh_real_space = np.abs(ifftshift(fftn(fftshift(tmp))))
            np.save(save_dir+'recon_'+self.scan_num+'_'+self.sign+'_coh', self.coh_real_space)

        if self.position_correction_flag_ini:
            np.save(save_dir+'points_'+self.scan_num+'_'+self.sign, self.points)
            np.save(save_dir+'points_'+self.scan_num+'_'+self.sign+'_ini', self.points_ini)
            np.save(save_dir+'points_list_'+self.scan_num+'_'+self.sign, self.points_list)

        if self.recon_code is not None:
            code_path, code_fn = os.path.split(self.recon_code)

            shutil.copy2(self.recon_code, save_dir+code_fn+'_'+self.sign+'.py')


        if self.mode_flag:
            orth.orthonormalize(self.scan_num, self.sign, 'probe')
            orth.orthonormalize(self.scan_num, self.sign, 'object')
            rm_phase_ramp(self.scan_num, self.sign,'mode')
            #rm_phase_ramp_mode(self.scan_num, self.sign)
            # plt.clf()
        elif self.multislice_flag:
            rm_phase_ramp(self.scan_num, self.sign,'ms')
        else:
            rm_phase_ramp(self.scan_num, self.sign)
            #save_hue(self.scan_num, self.sign)
            # plt.clf()

    def display_recon(self):

        disp_x = self.nx_obj - self.obj_pad - self.nx_prb
        disp_y = self.ny_obj - self.obj_pad - self.ny_prb
        disp_x_s = self.nx_obj//2 - disp_x//2
        disp_y_s = self.ny_obj//2 - disp_y//2
        if self.mode_flag:
            if self.prb_mode_num >= 4:
                nn_prb = 4
            else:
                nn_prb = self.prb_mode_num

            plt.figure()
            for ii in range(nn_prb):
                plt.subplot(2, 2, ii+1)
                plt.imshow(np.flipud(np.abs(self.prb_mode_ave[:,:, ii].T)))

            if self.obj_mode_num >= 4:
                nn_obj = 4
            else:
                nn_obj = self.obj_mode_num

            plt.figure()
            for ii in range(nn_obj):
                plt.subplot(2, 2, ii+1)
                plt.imshow(np.flipud(np.abs(self.obj_mode_ave[:,:, ii].T)))

        elif self.pc_flag:
            plt.figure()
            plt.subplot(221)
            plt.imshow(np.flipud(np.abs(self.prb_ave.T)))
            plt.subplot(222)
            plt.imshow(np.flipud(np.abs(self.obj_ave.T)))
            plt.subplot(223)
            plt.imshow(np.flipud(np.abs(self.coh_real_space.T)))
            plt.subplot(224)
            x_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.x_roi * self.ccd_pixel_um)
            y_real_space_pixel_nm = self.lambda_nm * self.z_m * 1.e6 / (self.y_roi * self.ccd_pixel_um)
            x_display_um = (np.arange(self.nx_prb) - self.nx_prb/2) * x_real_space_pixel_nm / 1.e3
            y_display_um = (np.arange(self.ny_prb) - self.ny_prb/2) * y_real_space_pixel_nm / 1.e3
            plt.plot(x_display_um, np.abs(self.coh_real_space[:, self.ny_prb//2]))
            plt.plot(y_display_um, np.abs(self.coh_real_space[self.nx_prb//2,:]))
            # plt.axis([-8,8,0,1])

        elif self.multislice_flag:
            if self.slice_num >= 4:
                nn_prb = 4
                nn_obj = 4
            else:
                nn_prb = self.slice_num
                nn_obj = self.slice_num

            plt.figure()
            for ii in range(nn_prb):
                plt.subplot(2, 2, ii+1)
                plt.imshow(np.flipud(np.abs(self.prb_ms_ave[0][ii].T)))

            plt.figure()
            for ii in range(nn_obj):
                plt.subplot(2, 2, ii+1)
                plt.imshow(np.flipud(np.abs(self.obj_ms_ave[:,:, ii].T)))
        else:
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.flipud(np.abs(self.prb_ave.T)))
            plt.subplot(122)
            plt.imshow(np.flipud(np.angle(self.obj_ave[disp_x_s:disp_x_s+disp_x,disp_y_s:disp_y_s+disp_y].T)),interpolation='none',cmap='bone')

        save_pic_dir = './recon_result/S'+self.scan_num+'/'+self.sign+'/recon_pic/'
        if not os.path.exists(save_pic_dir):
            os.makedirs(save_pic_dir)
        '''
        pic_fn = save_pic_dir+'recon_'+self.scan_num+'_'+self.sign+'.png'
        plt.savefig(pic_fn)
        shutil.copy2(pic_fn, '/home/klauer/laser_ptycho/fit_test/'+self.scan_num+'_'+self.sign+'.png')
        plt.show()
        '''
    def _run_parallel_over_points(self, function, *args, **kwargs):
        global main_pool
        #point_range = range(len(self.point_info))
        point_range = range(len(self.points[0]))
        p_range = range(len(self.point_info))

        if self.processes < 1:
            return [function(i, *args, **kwargs) for i in p_range]
        else:
            pool = None
            try:
                exit_event = mp.Event()
                pool = mp.Pool(processes=self.processes, initargs=(exit_event, ))
                main_pool = pool

                results = [pool.apply_async(parallel_function, [function.__name__, i] + list(args), kwargs)
                           for i in point_range]

                pool.close()

                alive = True
                while alive and not exit_event.is_set():
                    alive = any(p.is_alive() for p in pool._pool)
                    time.sleep(0.01)

                if exit_event.is_set():
                    print('** main exit event')
                else:
                    pool.join()

                results = [result.get() for result in results]
                if None in results:
                    raise Exception('Parallel failed')
                else:
                    return results

            except KeyboardInterrupt:
                print('Exiting')
                if pool is not None:
                    pool.terminate()
                    pool.join()
                sys.exit(1)

            except Exception as ex:
                print('Parallel execution failed: (%s) %s' % (ex.__class__.__name__, ex))
                traceback.print_exc()

                if pool is not None:
                    pool.terminate()
                    pool.join()
                else:
                    print('(Pool not yet initialized)')
                sys.exit(1)

    def check_file_num(self):
        return len([name for name in os.listdir(self.file_dir) if name.startswith(self.scan_num)])

    def load_tiff(self):

        num_file = len([filename for filename in os.listdir(self.file_dir) if filename.startswith(self.scan_num)])
        ##self.diff_array = []
        x_start = self.x_c-self.nx_prb/2
        y_start = self.y_c-self.ny_prb/2
        x_end = self.x_c+self.nx_prb/2
        y_end = self.y_c+self.ny_prb/2

        for i in range(self.file_num,num_file):
            fn = '%.6d'%(i)
            for parf in os.listdir(self.file_dir):
                if parf.startswith(self.scan_num+'_'+fn):
                    filename = parf
                    print(filename)
            tmp = tf.imread(self.file_dir+filename)
            #tmp[65:75,:] = 0.
            #tmp[:,180:190] = 0.
            tmp[85:95,:] = 0.
            tmp[:,160:170] = 0.
            tmp[:100,210:] = 0.
            tmp = tmp[:256,:256]
            #tmp = np.fliplr(tmp.T)
            #tmp = t[0:256,0:256]
            tmp = np.flipud(np.fliplr(tmp.T))
            #tmp[15:25,:] = 0.
            #tmp[:,235:245] = 0.
            #tmp[40:50,:] = 0.
            #tmp[:,200:210] = 0.
            tmp[tmp<=self.tif_threshold] = 0.
            self.diff_array.append(np.fft.fftshift(np.sqrt(tmp[x_start:x_end,y_start:y_end])))


    def load_tiff_timepix(self):

        num_file = len([filename for filename in os.listdir(self.file_dir) if filename.startswith(self.scan_num)])
        #self.diff_array = []
        x_start = self.x_c-self.nx_prb/2
        y_start = self.y_c-self.ny_prb/2
        x_end = self.x_c+self.nx_prb/2
        y_end = self.y_c+self.ny_prb/2

        for i in range(self.file_num,num_file):
            fn = '%.6d'%(i)
            for parf in os.listdir(self.file_dir):
                if parf.startswith(self.scan_num+'_'+fn):
                    filename = parf
                    print(filename)
            tmp = tf.imread(self.file_dir+filename)
            tmp[265:280,:] = 0.
            tmp[:,250:270] = 0.
            tmp = np.fliplr(tmp.T)
            nx_i,ny_i = np.shape(tmp)
            array_new = np.zeros((nx_i+4,ny_i+4))
            array_new[0:nx_i/2-1,0:ny_i/2-1] = tmp[0:nx_i/2-1,0:ny_i/2-1]
            array_new[nx_i/2+5:nx_i+4,0:ny_i/2-1] = tmp[nx_i/2+1:nx_i,0:ny_i/2-1]
            array_new[0:nx_i/2-1,ny_i/2+5:ny_i+4] = tmp[0:nx_i/2-1,ny_i/2+1:ny_i]
            array_new[nx_i/2+5:nx_i+4,ny_i/2+5:ny_i+4] = tmp[nx_i/2+1:nx_i,ny_i/2+1:ny_i]

            for ii in range(ny_i):
                array_new[nx_i/2-1:nx_i/2+2,i] = tmp[nx_i/2-1,i] / 3.
                array_new[nx_i/2+2:nx_i/2+5,i] = tmp[nx_i/2,i] / 3.

            for ii in range(nx_i):
                array_new[i,ny_i/2-1:ny_i/2+2] = tmp[i,ny_i/2-1] / 3.
                array_new[i,ny_i/2+2:ny_i/2+5] = tmp[i,ny_i/2] / 3.

            array_new[nx_i/2-1:nx_i/2+2,ny_i/2-1:ny_i/2+2] = tmp[nx_i/2-1,ny_i/2-1] / 9.
            array_new[nx_i/2-1:nx_i/2+2,ny_i/2+2:ny_i/2+5] = tmp[nx_i/2-1,ny_i/2] / 9.
            array_new[nx_i/2+2:nx_i/2+5,ny_i/2-1:ny_i/2+2] = tmp[nx_i/2,ny_i/2-1] / 9.
            array_new[nx_i/2+2:nx_i/2+5,ny_i/2+2:ny_i/2+5] = tmp[nx_i/2,ny_i/2] / 9.

            array_new[array_new<=self.tif_threshold] = 0.
            self.diff_array.append(np.fft.fftshift(np.sqrt(array_new[x_start:x_end,y_start:y_end])))


    # ptycho reconstruction
    @profile
    def recon_ptycho(self):

        self.x_pixel_m = self.lambda_nm * self.z_m * 1.e-3 / (self.x_roi * self.ccd_pixel_um)
        self.y_pixel_m = self.lambda_nm * self.z_m * 1.e-3 / (self.y_roi * self.ccd_pixel_um)
        print(self.x_pixel_m, self.y_pixel_m)

        if self.cal_scan_pattern_flag:
            if self.mesh_flag:
                self.cal_scan_pattern_mesh()
            elif self.fermat_flag:
                self.cal_scan_pattern_fermat()
            else:
                self.cal_scan_pattern()
        else:
            if self.bragg_flag:
                self.convert_scan_pattern()
            self.points[0,:] = np.round(self.points[0,:]*1.e-6/self.x_pixel_m)
            self.points[1,:] = np.round(self.points[1,:]*1.e-6/self.y_pixel_m)
            self.points[0,:] = self.points[0,:] - np.min(self.points[0,:]) + self.nx_prb / 2 + 15
            self.points[1,:] = self.points[1,:] - np.min(self.points[1,:]) + self.ny_prb / 2 + 15

        np.save('points_test', self.points)
        self.cal_obj_prb_dim()
        if(self.init_prb_flag):
            self.init_prb()
        if(self.init_obj_flag):
            if self.init_obj_dpc_flag:
                self.init_obj_stxm_dpc()
            else:
                self.init_obj()

        self.init_product()

        if self.pc_flag:
            if self.init_coh_flag:
                self.init_coh()
            if self.init_pc_filter_flag:
                self.init_pc_filter()
        '''
        if self.online_flag:
            file_num = self.check_file_num()
            while file_num < 10:
                file_num = self.check_file_num()
                time.sleep(5)
        '''

        if self.gpu_flag:
            self.gpu_init()

        self.time_start = time.time()

        for it in range(self.n_iterations):
            self.current_it = it
            '''
            if self.online_flag:
                file_num = self.check_file_num()
                if file_num > self.file_num:
                    nnn = self.num_points
                    ##self.file_num = file_num
                    self.load_tiff_timepix()
                    self.file_num = file_num
                    self.num_points,nx,ny = np.shape(self.diff_array)
                    self.point_info = np.array([(int(self.points[0, iii] - self.nx_prb/2), int(self.points[0, iii] + self.nx_prb/2), \
                                                 int(self.points[1, iii] - self.ny_prb/2), int(self.points[1, iii] + self.ny_prb/2)) \
                        for iii in range(self.num_points)])
                    for i_app in range(self.num_points-nnn):
                        self.product.append(np.zeros((self.nx_prb,self.ny_prb)).astype(complex))
            '''

            t0=time.time()

            if self.mode_flag:
                self.prb_mode_old = self.prb_mode.copy()
                self.obj_mode_old = self.obj_mode.copy()
            elif self.multislice_flag:
                for ii in range(self.num_points):
                    for jj in range(self.slice_num):
                        self.prb_ms_old[ii][jj] = self.prb_ms[ii][jj].copy()
                        self.product_old[ii][jj] = self.product[ii][jj].copy()
                self.obj_ms_old = self.obj_ms.copy()
            else:
                self.prb_old = self.prb.copy()
                self.obj_old = self.obj.copy()
            if self.update_coh_flag:
                self.coh_old = self.coh.copy()

            if it >= (self.pc_start * self.n_iterations):
                self.pc_modulus_flag = True

            if self.multislice_flag:
                self.multislice_propagate_forward()

            t1 = time.time() 
            self.elaps[0] +=t1-t0

            if self.pc_flag:
                if self.pc_modulus_flag:
                    self.recon_dm_trans_pc()
                    self.cal_coh(it)
                else:
                    self.recon_dm_trans()
            elif self.mode_flag:
                if self.dm_version == 1:
                    self.recon_dm_trans_mode()
                else:
                    self.recon_dm_trans_mode_real()
            elif self.multislice_flag:
                self.recon_dm_trans_ms()
            else:
                if (self.alg2_flag != False) and (it > (self.alg_percentage*self.n_iterations)):
                    if self.alg2_flag == 'ML':
                        self.recon_ml_trans()
                        print('ML')
                    elif self.alg2_flag == 'DM':
                        if( self.gpu_flag ):
                            self.recon_dm_trans_gpu()
                        else:
                            self.recon_dm_trans()
                        print('DM')
                    elif self.alg2_flag == 'DM_real':
                        self.recon_dm_trans_real()
                        print('DM real')
                    elif self.alg2_flag == 'ER':
                        self.recon_er_trans()
                        print('ER')
                else:
                    if self.alg_flag == 'ML':
                        self.recon_ml_trans()
                        print('ML')
                    elif self.alg_flag == 'DM':
                        if( self.gpu_flag ):
                            self.recon_dm_trans_gpu()
                        else:
                            self.recon_dm_trans()
                        print('DM')
                    elif self.alg_flag == 'DM_real':
                        self.recon_dm_trans_real()
                        print('DM real')
                    elif self.alg_flag == 'ER':
                        self.recon_er_trans()
                        print('ER')

            t0 = time.time() 
            self.elaps[1] += t0-t1

            if self.mode_flag:
                if(it >= self.start_update_probe):
                    if(it >= self.start_update_object):
                        self.cal_object_trans_mode()
                        self.cal_probe_trans_mode()
                    else:
                        self.cal_probe_trans_mode()
                else:
                    if(it >= self.start_update_object):
                        self.cal_object_trans_mode()
            elif self.multislice_flag:
                '''
                if(it >= self.start_update_probe):
                    if(it >= self.start_update_object):
                        self.cal_object_trans_ms()
                        self.cal_probe_trans_ms()
                    else:
                        self.cal_probe_trans_ms()
                else:
                    if(it >= self.start_update_object):
                        self.cal_object_trans_ms()
                '''
                self.multislice_propagate_backward(it)
            else:
                if(it >= self.start_update_probe):
                    if(it >= self.start_update_object):
                        self.cal_object_trans()
                        if(self.gpu_flag) :
                            self.cal_probe_trans_gpu() 
                        else :
                            self.cal_probe_trans()
                    else:
                        self.cal_probe_trans()
                else:
                    if(it >= self.start_update_object):
                        self.cal_object_trans()

            if(self.position_correction_flag):
                if(it >= np.floor(self.position_correction_start) and np.mod(it, self.position_correction_step) == 0):
                    if self.mode_flag:
                        position_count = self.position_correction_mode()
                    else:
                        position_count = self.position_correction()
                    print(position_count, 'positions updated')
                    # self.cal_position_correction()
                    # if(position_count == 0 or it > np.floor(self.start_ave*self.n_iterations)):
                    if(position_count == 0):
                        self.position_correction_flag = False

            t1 = time.time() 
            self.elaps[2]+=t1-t0


            self.cal_obj_error(it)
            self.cal_prb_error(it)

            t0 = time.time() 
            self.elaps[3]+=t0-t1
            if(self.gpu_flag) :
                self.cal_chi_error_gpu(it)
            else :
                self.cal_chi_error(it)

            t1 = time.time() 
            self.elaps[4]+=t1-t0

            if(self.update_coh_flag):
                self.cal_coh_error(it)

            if it == np.floor(self.start_ave*self.n_iterations):
                if self.mode_flag:
                    self.obj_mode_ave = self.obj_mode.copy()
                    self.prb_mode_ave = self.prb_mode.copy()
                elif self.multislice_flag:
                    self.obj_ms_ave = self.obj_ms.copy()
                    for ii in range(self.num_points):
                        for jj in range(self.slice_num):
                            self.prb_ms_ave[ii][jj] = self.prb_ms[ii][jj].copy()
                else:
                    self.obj_ave = self.obj.copy()
                    self.prb_ave = self.prb.copy()
                self.ave_i = 1.

            if it > np.floor(self.start_ave*self.n_iterations):
                if self.mode_flag:
                    self.obj_mode_ave = self.obj_mode_ave + self.obj_mode
                    self.prb_mode_ave = self.prb_mode_ave + self.prb_mode
                elif self.multislice_flag:
                    self.obj_ms_ave = self.obj_ms_ave + self.obj_ms
                    for ii in range(self.num_points):
                        for jj in range(self.slice_num):
                            self.prb_ms_ave[ii][jj] = self.prb_ms_ave[ii][jj] + self.prb_ms[ii][jj]
                else:
                    self.obj_ave = self.obj_ave + self.obj
                    self.prb_ave = self.prb_ave + self.prb
                self.ave_i = self.ave_i + 1

            if self.mode_flag:
                print(it, 'object_chi=', self.error_obj_mode[it, :self.obj_mode_num], 'probe_chi=', self.error_prb_mode[it, :self.prb_mode_num], 'diff_chi=', self.error_chi[it])
            elif self.multislice_flag:
                print(it, 'object_chi=', self.error_obj_ms[it, :self.slice_num], 'probe_chi=', self.error_prb_ms[it], 'diff_chi=', self.error_chi[it])
            elif self.update_coh_flag:
                print(it, 'object_chi=', self.error_obj[it], 'probe_chi=', self.error_prb[it], 'diff_chi=', self.error_chi[it], 'coh_chi=', self.error_coh[it])
            else:
                print(it, 'object_chi=', self.error_obj[it], 'probe_chi=', self.error_prb[it], 'diff_chi=', self.error_chi[it])

            if self.save_tmp_pic_flag:
                if np.mod(it, 5) == 0:
                    save_tmp_pic_dir = './tmp_pic/'
                    if not os.path.exists(save_tmp_pic_dir):
                        os.makedirs(save_tmp_pic_dir)

                    if self.prb_mode_num >= 4:
                        nn_prb = 4
                    else:
                        nn_prb = self.prb_mode_num

                    if self.obj_mode_num >= 4:
                        nn_obj = 4
                    else:
                        nn_obj = self.obj_mode_num

                    plt.figure()
                    for ii in range(nn_prb):
                        plt.subplot(2, 2, ii+1)
                        plt.imshow(np.flipud(np.abs(self.prb_mode[:,:, ii].T)))
                        plt.subplot(2, 2, ii+6)
                        plt.imshow(np.flipud(np.abs(self.obj_mode[:,:, ii].T)))

                    plt.savefig(save_tmp_pic_dir+'recon_'+self.scan_num+'_'+self.sign+'_'+np.str(it)+'.png')

            if self.update_product_flag:
                if (it >= self.start_update_product):
                    if np.mod(it, 5) == 0:
                        self.init_product()

            t0 = time.time() 
            self.elaps[5]+=t0-t1


            if self.online_flag:
                if it >= 0 and (it % 2) == 0:
                    if self.mode_flag:
                        obj_tmp = self.obj_mode[:,:,0]
                        prb_tmp = self.prb_mode[:,:,0]
                        obj_amp_max = self.amp_max
                        obj_amp_min = self.amp_min
                        obj_pha_max = self.pha_max
                        obj_pha_min = self.pha_min
                    elif self.multislice_flag:
                        obj_tmp = self.obj_ms[:,:,self.slice_num-1]
                        obj_tmp_2 = self.obj_ms[:,:,0]
                        prb_tmp = self.prb_ms[0][0]
                        obj_amp_max = self.amp_max[self.slice_num-1]
                        obj_amp_min = self.amp_min[self.slice_num-1]
                        obj_pha_max = self.pha_max[self.slice_num-1]
                        obj_pha_min = self.pha_min[self.slice_num-1]
                        obj_amp_max_2 = self.amp_max[0]
                        obj_amp_min_2 = self.amp_min[0]
                        obj_pha_max_2 = self.pha_max[0]
                        obj_pha_min_2 = self.pha_min[0]
                    else:
                        obj_tmp = self.obj
                        prb_tmp = self.prb
                        obj_amp_max = self.amp_max
                        obj_amp_min = self.amp_min
                        obj_pha_max = self.pha_max
                        obj_pha_min = self.pha_min
                    plt.figure(0)
                    plt.ion()
                    plt.clf()
                    plt.subplot(331)
                    plt.plot(self.points[0,:self.num_points],self.points[1,:self.num_points],'go')
                    plt.xlim([np.min(self.points[0,:])-10,np.max(self.points[0,:])+10])
                    plt.ylim([np.min(self.points[1,:])-10,np.max(self.points[1,:])+10])
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.axis('off')
                    plt.title('scan path')
                    plt.subplot(334)
                    plt.imshow(np.fft.fftshift(np.log10(np.flipud(self.diff_array[self.num_points-1].T)+0.001)))
                    plt.title('latest pattern')
                    plt.axis('off')
                    plt.subplot(332)
                    plt.imshow(np.flipud(np.abs(prb_tmp.T)))
                    plt.title('prb amp')
                    plt.axis('off')
                    plt.subplot(335)
                    plt.imshow(np.flipud(np.angle(prb_tmp.T)))
                    plt.title('prb pha')
                    plt.axis('off')
                    plt.subplot(333)
                    plt.imshow(np.flipud(np.abs(obj_tmp[self.obj_pad//2:-self.obj_pad//2,self.obj_pad//2:-self.obj_pad//2].T)),\
                            interpolation='none',cmap='bone',clim=[obj_amp_min,obj_amp_max])
                    plt.title('obj amp')
                    plt.axis('off')
                    plt.subplot(336)
                    plt.imshow(np.flipud(np.angle(obj_tmp[self.obj_pad//2:-self.obj_pad//2,self.obj_pad//2:-self.obj_pad//2].T)),\
                            interpolation='none',cmap='bone',clim=[obj_pha_min,obj_pha_max])
                    plt.title('obj pha')
                    plt.axis('off')
                    plt.subplot(337)
                    plt.plot(self.error_chi[:it])
                    plt.xlim([0,self.n_iterations])
                    plt.title('diff err')
                    if self.multislice_flag:
                        plt.subplot(339)
                        plt.imshow(np.flipud(np.angle(obj_tmp_2[self.obj_pad//2:-self.obj_pad//2,self.obj_pad//2:-self.obj_pad//2].T)),\
                                interpolation='none',cmap='bone',clim=[obj_pha_min_2,obj_pha_max_2])
                        plt.title('obj pha 2')
                        plt.axis('off')
                    #plt.subplot(338)
                    #plt.plot(self.error_prb[:it])
                    #plt.xlim([0,self.n_iterations])
                    #plt.title('prb err')
                    #plt.subplot(339)
                    #plt.plot(self.error_obj[:it])
                    #plt.xlim([0,self.n_iterations])
                    #plt.title('obj err')
                    plt.draw()
                    plt.show(block=False)
                    plt.pause(0.1)
                    #plt.show()
                    if self.pc_flag:
                        plt.figure(1)
                        plt.clf()
                        plt.subplot(121)
                        plt.imshow(self.coh.T,interpolation='none')
                        plt.title('psf')
                        #coh_t = np.zeros((self.nx_prb/2,self.ny_prb/2))
                        #coh_t[self.nx_prb/4-self.pc_kernel_n/2:self.nx_prb/4+self.pc_kernel_n/2,\
                        #      self.ny_prb/4-self.pc_kernel_n/2:self.ny_prb/4+self.pc_kernel_n/2] = self.coh
                        #coh_fft = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(coh_t))))
                        coh_fft = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(self.coh))))
                        plt.subplot(122)
                        plt.imshow(coh_fft.T,interpolation='none')
                        plt.title('coh')
                        plt.draw()
                        plt.show(block=False)
                        #plt.draw()
                        #plt.show()


            t1 = time.time() 
            self.elaps[8] += t1-t0


        self.time_end = time.time()
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('object size:', self.nx_obj, 'x', self.ny_obj)
        print('probe size:', self.nx_prb, 'x', self.ny_prb)
        print('total scan points:', self.num_points)
        print(self.n_iterations, 'iterations take', self.time_end - self.time_start, 'sec')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')

        print('time elaps : ', self.elaps) 

        if self.mode_flag:
            self.obj_mode_ave = self.obj_mode_ave / self.ave_i
            self.prb_mode_ave = self.prb_mode_ave / self.ave_i
        elif self.multislice_flag:
            self.obj_ms_ave = self.obj_ms_ave / self.ave_i
            for i in range(self.num_points):
                for j in range(self.slice_num):
                    self.prb_ms_ave[i][j] = self.prb_ms_ave[i][j] / self.ave_i
        else:
            self.prb_ave = self.prb_ave / self.ave_i
            self.obj_ave = self.obj_ave / self.ave_i
