
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
        __global__ void chi_sum_block( cuDoubleComplex * prb_obj, double * diff, double* diff_sum_sq , double* buff, int scale, int offset, int B  )
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

        if (tid < blockDim.x-B) {sdata[tid] += sdata[tid + B] ;}
        for (unsigned int s=B/2; s>1; s>>=1)
        {
            if (tid < s) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        /*
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
        */

        if(tid ==0 ) buff[blockIdx.x]=sdata[0] + sdata[1] ; 
        //if(tid ==0 && blockIdx.x < 16) printf("buff[%d]=%f , %d \n",blockIdx.x,sdata[0], point ) ;

        
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
        __global__ void dm_cal_dev(cuDoubleComplex *fft_tmp, double * amp_tmp, double *diff, double* dev_d, double* dev_tmp , int  nx , double sigma1, int offset, int B )
        {
        extern __shared__ double sdata[] ; 
        
        unsigned int tid = threadIdx.x ;
        unsigned long idx=tid+blockDim.x*blockIdx.x ;
        unsigned long idx_diff = idx + offset * blockDim.x * nx ;
        //unsigned int i = blockIdx.x /nx ;
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

        // first sum  only  Dim-B threads sum
        // B is largest 2^power which is < block size 
        if (tid < blockDim.x-B) {sdata[tid] += sdata[tid + B] ;}

        //now just reduce first B items of sdata  
        for (unsigned int s=B/2; s>1; s>>=1)
        {
            if (tid < s ) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        /*
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
        */

        if ( tid == 0 ) 
        dev_tmp[blockIdx.x] = (sdata[0]+sdata[1])/scale ;
        
        }
        else 
        dev_tmp[blockIdx.x] = 0.0 ; 


        }
        }

        extern "C" {
        __global__ void dm_reduce_dev(double* dev_tmp,  double * power, int B   )
        {
        extern __shared__ double sdata[] ; 

        
        unsigned int tid = threadIdx.x ;
        unsigned int idx = tid + blockDim.x * blockIdx.x  ;
        sdata[tid] = dev_tmp[idx];
        __syncthreads();

        if (tid < blockDim.x-B) {sdata[tid] += sdata[tid + B] ;}
        for (unsigned int s=B/2; s>1; s>>=1)
        {
            if (tid < s) {
            sdata[tid] += sdata[tid + s];}
            __syncthreads();
        }
        /*
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
        */

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
        __global__ void obj_trans(cuDoubleComplex* product, cuDoubleComplex* obj, cuDoubleComplex * prb,   double * norm, int * point_info, 
          int o_ny , int i  )
        {
        unsigned int idx_pr = threadIdx.x + blockIdx.x * blockDim.x ;
        unsigned long idx_pd = idx_pr + i* blockDim.x*gridDim.x ;

        int xstart = point_info[i*4];
        int ystart = point_info[i*4+2];
        
        unsigned int idx_o =  threadIdx.x + (blockIdx.x  + xstart )*o_ny + ystart ;
        cuDoubleComplex p = prb[idx_pr] ;
        double p2 = cuCabs(p) * cuCabs(p) ;

        obj[idx_o] = cuCadd(obj[idx_o] , cuCmul(cuConj(p) , product[idx_pd] )) ;
        norm[idx_o] = norm[idx_o] + p2 ; 


        }
        }
        
        extern "C" {
        __global__ void zero(cuDoubleComplex* C,    double * D, double alpha, int  N ) 
        {
            unsigned long idx = threadIdx.x + blockIdx.x* blockDim.x ;
            if (idx < N ) 
            {
                C[idx] = make_cuDoubleComplex(0.0,0.0 ) ;
                D[idx] = alpha ;
            }
        }
        } 
    


        
