
/* --------------------------------------------------------------------------------------------------------
Function: Batchnorm (Aleternative to nn.Batchnorm1d()) for CPU

Author: Ramanarayan Mohanty (Intel Corp.) & MD Vasimuddin(Intel Corp.)

-----------------------------------------------------------------------------------------------------------
*/ 


#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <cmath> 
#include <tuple>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

# define eps 1e-7f


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batchnorm_fwd_impl(at::Tensor& input, at::Tensor tgamma, at::Tensor tbeta)//, at::Tensor& inNrm)
{
	
    int64_t n_input = input.size(0);
    int64_t n_feat = input.size(1); // Features;
     
	const float sqrt_eps = 1e-7f;

	const float recp_n = 1.0f/(n_input);

    auto tbmean = at::empty({n_feat}, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* bmean_fnl=tbmean.data_ptr<float>();

    auto tbvar = at::empty({n_feat}, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* bvar_fnl=tbvar.data_ptr<float>();

    auto tout = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* values_fnl=tout.data_ptr<float>();

    auto tinNrm = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* inpNorm_fnl=(float*)tinNrm.data_ptr();

    float * inp = (float *) input.data_ptr();

        
    int num_threads = omp_get_max_threads();
    float* sum_arr_ = (float*) _mm_malloc ( num_threads * n_feat * sizeof(float), 64);
    float* sumsq_arr_ = (float*) _mm_malloc (num_threads * n_feat *  sizeof(float), 64);   
    memset(sum_arr_, 0,  num_threads* n_feat * sizeof(float));
    memset(sumsq_arr_, 0,  num_threads * n_feat * sizeof(float));

	#pragma omp parallel 
    {
    	int num_threads = omp_get_num_threads();
    	int tid = omp_get_thread_num();
    	int chunksize = (n_input % num_threads == 0) ? (n_input / num_threads) : ((n_input / num_threads) + 1);
    	int thr_begin = (tid * chunksize < n_input) ? (tid * chunksize) : n_input;
    	int thr_end = ((tid + 1) * chunksize < n_input) ? ((tid+1) * chunksize) : n_input;
    	    // std::cout << "Number of Threads: " << num_threads << std::endl;


    	float *sum_arr = sum_arr_ + n_feat * tid;
    	float *sumsq_arr = sumsq_arr_ + n_feat * tid;

    	for ( int i = thr_begin; i < thr_end; ++i)
    	{
            int j=0;
            for (j=0 ; j < n_feat-16 + 1; j+=16)
            {
                // PLace Full mask code
                __m512 lcl_vsum = _mm512_loadu_ps((__m512*)& sum_arr[j] );
                __m512 lcl_vsumsq = _mm512_loadu_ps((__m512*)& sumsq_arr[j] );

                __m512 lcl_vinput = _mm512_loadu_ps((__m512*)& inp[i*n_feat + j] ); 
                
                lcl_vsum   = _mm512_add_ps( lcl_vsum, lcl_vinput ); 
                lcl_vsumsq = _mm512_add_ps( lcl_vsumsq, _mm512_mul_ps( lcl_vinput, lcl_vinput ) );  

                // print_m512Var(lcl_vsum);
                // print_m512Var(lcl_vsumsq); 

                _mm512_storeu_ps ((__m512*)&sum_arr[j], lcl_vsum);
                _mm512_storeu_ps ((__m512*)&sumsq_arr[j], lcl_vsumsq);
            }
       
            if(j < n_feat)
            {
                //  Place partial mask code
                __m512 src;
                __mmask16 Msk = 0xFFFF  >> (16 - (n_feat % 16)); 
                __m512 lcl_vsum_ = _mm512_mask_loadu_ps (src,  Msk, (__m512*)& sum_arr[j]);
                __m512 lcl_vsumsq_ = _mm512_mask_loadu_ps ( src,  Msk, (__m512*)& sumsq_arr[j]);


                __m512 lcl_vinput_ = _mm512_mask_loadu_ps ( src,  Msk, (__m512*)& inp[i*n_feat + j]);

                lcl_vsum_ = _mm512_mask_add_ps ( src,  Msk, lcl_vsum_, lcl_vinput_);

                __m512 lcl_vsq_ = _mm512_mask_mul_ps ( src,  Msk, lcl_vinput_, lcl_vinput_);
                lcl_vsumsq_ = _mm512_mask_add_ps ( src,  Msk, lcl_vsumsq_, lcl_vsq_);

                _mm512_mask_storeu_ps ((__m512*)&sum_arr[j],  Msk, lcl_vsum_);
                _mm512_mask_storeu_ps ((__m512*)&sumsq_arr[j],  Msk, lcl_vsumsq_);
            }

        }
    }

  
    // #pragma omp parallel for 
    for (int i = 0; i < num_threads-1; ++i)
    {
    	for (int j = 0; j < n_feat; ++j)
    	{
    		sum_arr_[j] = sum_arr_[j] + sum_arr_[ (i+1)*n_feat + j];
    		sumsq_arr_[j] = sumsq_arr_[j] + sumsq_arr_[ (i+1)*n_feat + j];
    	}
    }

        float* bgamma = (float *) tgamma.data_ptr();
        float* bbeta = (float *) tbeta.data_ptr();

        __m512 lcl_vsqrt_eps = _mm512_set1_ps(sqrt_eps);
        __m512 lcl_vrec_n  = _mm512_set1_ps(recp_n);
        __m512 lcl_vone    = _mm512_set1_ps(1.0);
      
        __m512 lcl_vgamma = _mm512_set1_ps(bgamma[0]);
        __m512 lcl_vbeta = _mm512_set1_ps(bbeta[0]);
  
    float* std = (float*) _mm_malloc (n_feat *  sizeof(float), 64);

       int v2=0;
    for ( v2 = 0; v2 < n_feat-16 + 1; v2+=16)
    {
            

            __m512 lcl_vo, lcl_inpNorm, lcl_vbmean, lcl_vbmeansq, lcl_vsqbmean, lcl_vbrstd, lcl_vvar;

            __m512 lcl_vsum = _mm512_loadu_ps((__m512*)& sum_arr_[v2] );
            __m512 lcl_vsumsq = _mm512_loadu_ps((__m512*)& sumsq_arr_[v2] );

            lcl_vbmean   = _mm512_mul_ps( lcl_vrec_n,   lcl_vsum   );   /* E(X) */
            lcl_vbmeansq = _mm512_mul_ps( lcl_vbmean,   lcl_vbmean );   /* E(X)^2 */
            lcl_vsqbmean = _mm512_mul_ps( lcl_vrec_n, lcl_vsumsq );   /* E(X^2) */
            lcl_vvar     = _mm512_sub_ps( lcl_vsqbmean, lcl_vbmeansq ); /* variance */

            lcl_vbrstd   = _mm512_div_ps( lcl_vone, _mm512_sqrt_ps( _mm512_add_ps( lcl_vvar, lcl_vsqrt_eps ) ) );

            _mm512_storeu_ps((__m512*)&bmean_fnl[v2], lcl_vbmean);
            _mm512_storeu_ps((__m512*)&bvar_fnl[v2], lcl_vvar);
            _mm512_storeu_ps((__m512*)&std[v2], lcl_vbrstd);
    }        
    if(v2 < n_feat)
    {
        __m512 src;
        __mmask16 Msk = 0xFFFF  >> (16 - (n_feat % 16)); 

        __m512 lcl_vo_, lcl_inpNorm_, lcl_vbmean_, lcl_vbmeansq_, lcl_vsqbmean_, lcl_vbrstd_, lcl_vvar_;

        __m512 lcl_vsum_ = _mm512_mask_loadu_ps (src,  Msk, (__m512*)& sum_arr_[v2]);
        __m512 lcl_vsumsq_ = _mm512_mask_loadu_ps ( src,  Msk, (__m512*)& sumsq_arr_[v2]);

        lcl_vbmean_   = _mm512_mask_mul_ps(src, Msk, lcl_vrec_n,   lcl_vsum_   );   /* E(X) */
        lcl_vbmeansq_ = _mm512_mask_mul_ps(src, Msk, lcl_vbmean_,   lcl_vbmean_ );   /* E(X)^2 */
        lcl_vsqbmean_ = _mm512_mask_mul_ps(src, Msk, lcl_vrec_n, lcl_vsumsq_ );   /* E(X^2) */
        lcl_vvar_     = _mm512_mask_sub_ps(src, Msk, lcl_vsqbmean_, lcl_vbmeansq_ ); /* variance */


        lcl_vbrstd_   = _mm512_mask_div_ps(src, Msk, lcl_vone, _mm512_mask_sqrt_ps(src, Msk, _mm512_mask_add_ps(src, Msk, lcl_vvar_, lcl_vsqrt_eps ) ) );

        _mm512_mask_storeu_ps( (__m512*)&bmean_fnl[v2], Msk, lcl_vbmean_);
        _mm512_mask_storeu_ps( (__m512*)&bvar_fnl[v2], Msk, lcl_vvar_);
        _mm512_mask_storeu_ps( (__m512*)&std[v2], Msk, lcl_vbrstd_);
    }
   

    #pragma omp parallel 
    {
    	int num_threads = omp_get_num_threads();
    	int tid = omp_get_thread_num();
    	int chunksize = (n_input % num_threads == 0) ? (n_input / num_threads) : ((n_input / num_threads) + 1);
    	int thr_begin = (tid * chunksize < n_input) ? (tid * chunksize) : n_input;
    	int thr_end = ((tid + 1) * chunksize < n_input) ? ((tid+1) * chunksize) : n_input;

        for (int v1=thr_begin; v1 < thr_end; v1++)
        {
            int v2 = 0;
            for (v2 = 0; v2 < n_feat-16 + 1; v2+=16)
            {

                __m512 lcl_vo, lcl_inpNorm;

                __m512 lcl_vsum = _mm512_loadu_ps((__m512*)& sum_arr_[v2] );
                __m512 lcl_vsumsq = _mm512_loadu_ps((__m512*)& sumsq_arr_[v2] );

                __m512 lcl_vbmean = _mm512_loadu_ps((__m512*)& bmean_fnl[v2] );
                __m512 lcl_vbrstd = _mm512_loadu_ps((__m512*)& std[v2] );
                __m512 lcl_vinput_ptr = _mm512_loadu_ps((__m512*)& inp[v1 * n_feat + v2] );
                /* BN + scale (gamma, beta) */
                lcl_inpNorm = _mm512_sub_ps( lcl_vinput_ptr, lcl_vbmean );
                lcl_inpNorm = _mm512_mul_ps(lcl_inpNorm, lcl_vbrstd);
                lcl_vo = _mm512_fmadd_ps( lcl_inpNorm, lcl_vgamma, lcl_vbeta );
                _mm512_storeu_ps (( __m512*)&inpNorm_fnl[v1 * n_feat + v2] , lcl_inpNorm) ; 
                _mm512_storeu_ps (( __m512*)&values_fnl[v1 * n_feat + v2] , lcl_vo) ;
                
            }
	if(v2 < n_feat)
            {
                __m512 src;
                __mmask16 Msk = 0xFFFF  >> (16 - (n_feat % 16)); 

                __m512 lcl_vo_, lcl_inpNorm_;

                __m512 lcl_vsum_ = _mm512_mask_loadu_ps (src,  Msk, (__m512*)& sum_arr_[v2]);
                __m512 lcl_vsumsq_ = _mm512_mask_loadu_ps ( src,  Msk, (__m512*)& sumsq_arr_[v2]);

                
                __m512 lcl_vbmean_ = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& bmean_fnl[v2] );
                __m512 lcl_vbrstd_ = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& std[v2] );

                __m512 lcl_vinput_ptr_ = _mm512_mask_loadu_ps(src, Msk, (__m512*)& inp[v1 * n_feat + v2] );
                /* BN + scale (gamma, beta) */
                lcl_inpNorm_ = _mm512_mask_sub_ps(src, Msk, lcl_vinput_ptr_, lcl_vbmean_ );
                lcl_inpNorm_ = _mm512_mask_mul_ps(src, Msk, lcl_inpNorm_, lcl_vbrstd_);
                lcl_vo_ = _mm512_mask_fmadd_ps(lcl_inpNorm_, Msk, lcl_vgamma, lcl_vbeta );

                _mm512_mask_storeu_ps (( __m512*)&inpNorm_fnl[v1 * n_feat + v2] , Msk, lcl_inpNorm_) ; 
                _mm512_mask_storeu_ps (( __m512*)&values_fnl[v1 * n_feat + v2] , Msk, lcl_vo_) ;
                
            }
        }
    }

    
    _mm_free(std);
    _mm_free(sum_arr_);
    _mm_free(sumsq_arr_);

    // print_matrix(inpNorm_, n_input, n_feat, 1);

	return {tout, tinNrm, tbmean, tbvar};
}







std::tuple <at::Tensor, at::Tensor, at::Tensor>
batchnorm_bwd_impl(at::Tensor& grad, at::Tensor& tinput, at::Tensor tinpNorm, at::Tensor tbmean, at::Tensor tbvar, at::Tensor& tgamma, at::Tensor& tbeta)
{
    
	int64_t n_input = tinput.size(0);
    int64_t n_feat = tinput.size(1); // Features;
    
    
    float* pgamma = (float *) tgamma.data_ptr();
    float gamma_ = pgamma[0];
    float* pbeta = (float *) tbeta.data_ptr();
    float beta_ = pbeta[0];

    auto grad_inp = at::empty_like(tinput, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* dinp_val=grad_inp.data_ptr<float>();

    auto grad_gamma = at::empty_like(tgamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* gamma_fnl = grad_gamma.data_ptr<float>();

    auto grad_beta = at::empty_like(tbeta, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    float* beta_fnl = (float*)grad_beta.data_ptr();

    // int64_t nBatch = 1;

    const float sqrt_eps = 1e-7f;
    const float recp_n = 1.0f/(n_input);

    __m512 lcl_vgamma = _mm512_set1_ps(gamma_);
    __m512 lcl_vbeta = _mm512_set1_ps(beta_);


    // Load the tensors (output gradient, input and inputNorm)
    float * dgrad = (float *) grad.data_ptr();
    float * inp = (float *) tinput.data_ptr();
    float * inp_norm = (float *) tinpNorm.data_ptr();

    __m512 dgamma, dbeta, dinp_norm, dinp,  dinp_, dinp_norm_;

    // Initialize the grad_gamma and and grad_beta to zero
    
    __m512 lcl_dgamma_ = _mm512_setzero_ps();
    __m512 lcl_dbeta_ = _mm512_setzero_ps();

    // Load the mean and variance to the vector intrinsics;
    float * bmean = (float *) tbmean.data_ptr();
    float * bvar = (float *) tbvar.data_ptr();

    
    float* x_mu = (float*) _mm_malloc (n_input * n_feat * sizeof(float), 64);
    float* dx_norm = (float*) _mm_malloc (n_input * n_feat * sizeof(float), 64);
    float* std_inv = (float*) _mm_malloc (n_feat * sizeof(float), 64);
    float* neg_std_inv = (float*) _mm_malloc (n_feat * sizeof(float), 64);
    float* cube_std_inv = (float*) _mm_malloc (n_feat * sizeof(float), 64);

    	// Intialize the vectors by broadcasting with the constants and compute the standard deviation with epsilon
    __m512 lcl_vsqrt_eps = _mm512_set1_ps(sqrt_eps);
    __m512 lcl_vrec_n  = _mm512_set1_ps(recp_n);
    __m512 lcl_vone    = _mm512_set1_ps(1.0);
    __m512 lcl_vnegone    = _mm512_set1_ps(-1.0);
    __m512 lcl_vneghalf    = _mm512_set1_ps(-0.5); 

     int v2 = 0;
    for (v2 = 0; v2 < n_feat-16 + 1; v2 +=16)
    {
        __m512 lcl_vvar = _mm512_loadu_ps((__m512*)& bvar[v2]);
        // Compute square root of variance        
        __m512 lcl_vvar_eps_sqrt = _mm512_sqrt_ps(_mm512_add_ps( lcl_vvar, lcl_vsqrt_eps));
        // 1/sqrt(var^2 + eps)
        __m512 lcl_vstd_inv   = _mm512_div_ps( lcl_vone, lcl_vvar_eps_sqrt);  
        // -1/sqrt(var^2 + eps)
        __m512 lcl_vneg_std_inv   = _mm512_mul_ps( lcl_vnegone, lcl_vstd_inv); 

        __m512 lcl_vcube_std_inv = _mm512_mul_ps(lcl_vneghalf, _mm512_mul_ps(lcl_vstd_inv, _mm512_mul_ps(lcl_vstd_inv, lcl_vstd_inv)));   

        _mm512_storeu_ps((__m512*)&cube_std_inv[v2], lcl_vcube_std_inv);
        _mm512_storeu_ps((__m512*)&neg_std_inv[v2], lcl_vneg_std_inv);
        _mm512_storeu_ps((__m512*)&std_inv[v2], lcl_vstd_inv);
    }
    if(v2 < n_feat)
    
        {
            __m512 src;
            __mmask16 Msk = 0xFFFF  >> (16 - n_feat % 16); 

            __m512 lcl_vvar = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& bvar[v2]);
            
            // Compute square root of variance        
            __m512 lcl_vvar_eps_sqrt = _mm512_mask_sqrt_ps(src,  Msk, _mm512_mask_add_ps(src,  Msk, lcl_vvar, lcl_vsqrt_eps));
            // 1/sqrt(var^2 + eps)
            __m512 lcl_vstd_inv   = _mm512_mask_div_ps(src,  Msk, lcl_vone, lcl_vvar_eps_sqrt);  
            // -1/sqrt(var^2 + eps)
            __m512 lcl_vneg_std_inv   = _mm512_mask_mul_ps(src,  Msk, lcl_vnegone, lcl_vstd_inv); 

            __m512 lcl_vcube_std_inv = _mm512_mask_mul_ps(src,  Msk, lcl_vneghalf, _mm512_mask_mul_ps(src,  Msk, lcl_vstd_inv, _mm512_mask_mul_ps(src,  Msk, lcl_vstd_inv, lcl_vstd_inv)));   
			
			_mm512_mask_storeu_ps((__m512*)&cube_std_inv[v2], Msk, lcl_vcube_std_inv);
            _mm512_mask_storeu_ps((__m512*)&neg_std_inv[v2], Msk, lcl_vneg_std_inv);
            _mm512_mask_storeu_ps((__m512*)&std_inv[v2], Msk, lcl_vstd_inv);
		}

    #pragma omp parallel
    {
    	int num_threads = omp_get_num_threads();
    	int tid = omp_get_thread_num();
    	int chunksize = (n_input % num_threads == 0) ? (n_input / num_threads) : ((n_input / num_threads) + 1);
    	int thr_begin = (tid * chunksize < n_input) ? (tid * chunksize) : n_input;
    	int thr_end = ((tid + 1) * chunksize < n_input) ? ((tid+1) * chunksize) : n_input;

	    for (int v1=thr_begin; v1 < thr_end; v1++)
	    {
	        int v2 = 0;
	        for (v2 = 0; v2 < n_feat-16 + 1; v2 +=16)
	        {
	        	__m512 lcl_x = _mm512_loadu_ps((__m512*)& inp[v1*n_feat + v2]);
	            __m512 lcl_vbmean = _mm512_loadu_ps((__m512*)& bmean[v2]);
	           
	            __m512 lcl_dgrad = _mm512_loadu_ps((__m512*)& dgrad[v1*n_feat + v2]);

	            __m512 lcl_x_mu = _mm512_sub_ps(lcl_x, lcl_vbmean);

	            __m512 lcl_dx_norm = _mm512_mul_ps(lcl_dgrad, lcl_vgamma);

	            _mm512_storeu_ps (( __m512*)&x_mu[v1 * n_feat + v2] , lcl_x_mu) ;
	            _mm512_storeu_ps (( __m512*)&dx_norm[v1 * n_feat + v2] , lcl_dx_norm) ;
	           

	        }
		if(v2 < n_feat)
	    
	        {
	            __m512 src;
	            __mmask16 Msk = 0xFFFF  >> (16 - n_feat % 16); 

	            __m512 lcl_vbmean = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& bmean[v2]);
   	            __m512 lcl_x = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& inp[v1*n_feat + v2]);

	            __m512 lcl_dgrad = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dgrad[v1*n_feat + v2]);

	            __m512 lcl_x_mu = _mm512_mask_sub_ps(src,  Msk, lcl_x, lcl_vbmean);

	            __m512 lcl_dx_norm = _mm512_mask_mul_ps(src,  Msk, lcl_dgrad, lcl_vgamma);

	            _mm512_mask_storeu_ps (( __m512*)&x_mu[v1 * n_feat + v2] , Msk, lcl_x_mu) ;
	            _mm512_mask_storeu_ps (( __m512*)&dx_norm[v1 * n_feat + v2] , Msk, lcl_dx_norm) ;
	           
	        }
	    }
	}

    int num_threads = omp_get_max_threads();
    float* dvar_sum_ = (float*) _mm_malloc (num_threads * n_feat * sizeof(float), 64);
    float* dxnorm_stdinv_sum_ = (float*) _mm_malloc (num_threads * n_feat * sizeof(float), 64);
    float* x_mu_sum_ = (float*) _mm_malloc (num_threads * n_feat * sizeof(float), 64);

    memset(dvar_sum_, 0, num_threads * n_feat * sizeof(float));
    memset(dxnorm_stdinv_sum_, 0, num_threads * n_feat * sizeof(float));
    memset(x_mu_sum_, 0, num_threads * n_feat * sizeof(float));

   
    #pragma omp parallel
    {
    	int num_threads = omp_get_num_threads();
    	int tid = omp_get_thread_num();
    	int chunksize = (n_input % num_threads == 0) ? (n_input / num_threads) : ((n_input / num_threads) + 1);
    	int thr_begin = (tid * chunksize < n_input) ? (tid * chunksize) : n_input;
    	int thr_end = ((tid + 1) * chunksize < n_input) ? ((tid+1) * chunksize) : n_input;

    	float *dvar_sum = dvar_sum_ + n_feat * tid;
    	float *dxnorm_stdinv_sum = dxnorm_stdinv_sum_ + n_feat * tid;
    	float *x_mu_sum = x_mu_sum_ + n_feat * tid;

	    for (int v1=thr_begin; v1 < thr_end; v1++)
	    {
            int v2 = 0;
            for (v2 = 0; v2 < n_feat - 16 + 1; v2+=16)
            {
                __m512 lcl_dvar_sum = _mm512_loadu_ps((__m512*)& dvar_sum[v2] );
                __m512 lcl_x_mu_sum = _mm512_loadu_ps((__m512*)& x_mu_sum[v2] );
                __m512 lcl_dxnorm_stdinv_sum = _mm512_loadu_ps((__m512*)& dxnorm_stdinv_sum[v2]);

                __m512 lcl_dx_norm = _mm512_loadu_ps((__m512*)& dx_norm[v1 * n_feat + v2]);
                __m512 lcl_x_mu = _mm512_loadu_ps((__m512*)& x_mu[v1 * n_feat + v2]);
                __m512 lcl_vneg_std_inv = _mm512_loadu_ps((__m512*)& neg_std_inv[v2]);

                lcl_dvar_sum = _mm512_add_ps(lcl_dvar_sum, _mm512_mul_ps(lcl_dx_norm, lcl_x_mu)); 

                lcl_dxnorm_stdinv_sum = _mm512_add_ps(lcl_dxnorm_stdinv_sum, _mm512_mul_ps(lcl_dx_norm, lcl_vneg_std_inv));

                lcl_x_mu_sum = _mm512_add_ps(lcl_x_mu_sum, lcl_x_mu);

                _mm512_storeu_ps((__m512*)&x_mu_sum[v2], lcl_x_mu_sum);
                _mm512_storeu_ps((__m512*)&dvar_sum[v2], lcl_dvar_sum);
                _mm512_storeu_ps((__m512*)&dxnorm_stdinv_sum[v2], lcl_dxnorm_stdinv_sum);
            }

         if(v2 < n_feat)
        
            {
                __m512 src;
                __mmask16 Msk = 0xFFFF  >> (16 - n_feat % 16);                
    
                __m512 lcl_dvar_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dvar_sum[v2] );
                __m512 lcl_x_mu_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& x_mu_sum[v2] );
                __m512 lcl_dxnorm_stdinv_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dxnorm_stdinv_sum[v2]);

                __m512 lcl_dx_norm = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dx_norm[v1 * n_feat + v2]);
                __m512 lcl_x_mu = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& x_mu[v1 * n_feat + v2]);
                __m512 lcl_vneg_std_inv = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& neg_std_inv[v2]);

                lcl_dvar_sum = _mm512_mask_add_ps(src,  Msk, lcl_dvar_sum, _mm512_mask_mul_ps(src,  Msk, lcl_dx_norm, lcl_x_mu)); 

                lcl_dxnorm_stdinv_sum = _mm512_mask_add_ps(src,  Msk, lcl_dxnorm_stdinv_sum, _mm512_mask_mul_ps(src,  Msk, lcl_dx_norm, lcl_vneg_std_inv));

                lcl_x_mu_sum = _mm512_mask_add_ps(src,  Msk, lcl_x_mu_sum, lcl_x_mu);

                _mm512_mask_storeu_ps((__m512*)&x_mu_sum[v2], Msk, lcl_x_mu_sum);
                _mm512_mask_storeu_ps((__m512*)&dvar_sum[v2], Msk, lcl_dvar_sum);
                _mm512_mask_storeu_ps((__m512*)&dxnorm_stdinv_sum[v2], Msk, lcl_dxnorm_stdinv_sum);
        	}
        }
	}

	for (int i = 0; i < num_threads-1; ++i)
    {
    	for (int j = 0; j < n_feat; ++j)
    	{
    		dvar_sum_[j] = dvar_sum_[j] + dvar_sum_[ (i+1)*n_feat + j];
    		dxnorm_stdinv_sum_[j] = dxnorm_stdinv_sum_[j] + dxnorm_stdinv_sum_[ (i+1)*n_feat + j];
    		x_mu_sum_[j] = x_mu_sum_[j] + x_mu_sum_[ (i+1)*n_feat + j];
    	}
    }

	__m512 lcl_vnegtwo = _mm512_set1_ps(-2.0);
    __m512 lcl_vtwo = _mm512_set1_ps(2.0);

    // __m512 lcl_dgamma = _mm512_setzero_ps();
    // __m512 lcl_dbeta = _mm512_setzero_ps();

    float* dvar = (float*) _mm_malloc (n_feat * sizeof(float), 64);
    float* dmu = (float*) _mm_malloc (n_feat * sizeof(float), 64);

    for (v2 = 0; v2 < n_feat-16 + 1; v2+=16)
    {
        __m512 lcl_vstd_inv = _mm512_loadu_ps((__m512*)& std_inv[v2]);
        __m512 lcl_vcube_std_inv = _mm512_loadu_ps((__m512*)& cube_std_inv[v2]);
        __m512 lcl_dvar_sum = _mm512_loadu_ps((__m512*)& dvar_sum_[v2]);

        __m512 lcl_dvar = _mm512_mul_ps(lcl_dvar_sum, lcl_vcube_std_inv);

        __m512 lcl_dxnorm_stdinv_sum = _mm512_loadu_ps((__m512*)& dxnorm_stdinv_sum_[v2]);
        __m512 lcl_x_mu_sum = _mm512_loadu_ps((__m512*)& x_mu_sum_[v2]);
        __m512 lcl_dmu_tmp1 = _mm512_mul_ps(lcl_dvar, _mm512_mul_ps(lcl_vnegtwo, _mm512_mul_ps(lcl_vrec_n, lcl_x_mu_sum)));
        __m512 lcl_dmu = _mm512_add_ps(lcl_dxnorm_stdinv_sum, lcl_dmu_tmp1);
        _mm512_storeu_ps (( __m512*)&dvar[v2] , lcl_dvar) ;
        _mm512_storeu_ps (( __m512*)&dmu[v2] , lcl_dmu) ;
    }
    if(v2 < n_feat)        
    {
        __m512 src;
        __mmask16 Msk = 0xFFFF  >> (16 - n_feat % 16); 
        __m512 lcl_vstd_inv = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& std_inv[v2]);
        __m512 lcl_vcube_std_inv = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& cube_std_inv[v2]);
        __m512 lcl_dvar_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dvar_sum_[v2]);

        __m512 lcl_dvar = _mm512_mask_mul_ps(src,  Msk, lcl_dvar_sum, lcl_vcube_std_inv);

        __m512 lcl_dxnorm_stdinv_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dxnorm_stdinv_sum_[v2]);
        __m512 lcl_x_mu_sum = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& x_mu_sum_[v2]);
        __m512 lcl_dmu_tmp1 = _mm512_mask_mul_ps(src,  Msk, lcl_dvar, _mm512_mask_mul_ps(src,  Msk, lcl_vnegtwo, _mm512_mask_mul_ps(src,  Msk, lcl_vrec_n, lcl_x_mu_sum)));
        __m512 lcl_dmu = _mm512_mask_add_ps(src,  Msk, lcl_dxnorm_stdinv_sum, lcl_dmu_tmp1);

        _mm512_mask_storeu_ps (( __m512*)&dvar[v2] , Msk, lcl_dvar) ;
        _mm512_mask_storeu_ps (( __m512*)&dmu[v2] , Msk, lcl_dmu) ;
    }

    float* gamma_val_ = (float*) _mm_malloc (num_threads * n_feat * sizeof(float), 64);
    float* beta_val_ = (float*) _mm_malloc (num_threads * n_feat * sizeof(float), 64);
    memset(gamma_val_, 0, num_threads * n_feat * sizeof(float));
    memset(beta_val_, 0, num_threads * n_feat * sizeof(float));

    #pragma omp parallel
    {
    	int num_threads = omp_get_num_threads();
    	int tid = omp_get_thread_num();
    	int chunksize = (n_input % num_threads == 0) ? (n_input / num_threads) : ((n_input / num_threads) + 1);
    	int thr_begin = (tid * chunksize < n_input) ? (tid * chunksize) : n_input;
    	int thr_end = ((tid + 1) * chunksize < n_input) ? ((tid+1) * chunksize) : n_input;

    	float * gamma_val = gamma_val_ + tid * n_feat;
    	float * beta_val = beta_val_ + tid * n_feat;

    	for (int v1=thr_begin; v1 < thr_end; v1++)
        {
            int v2 = 0;
            for (v2 = 0; v2 < n_feat-16 + 1; v2+=16)
            {
                __m512 lcl_dgamma = _mm512_loadu_ps((__m512*)& gamma_val_[v2] );
                __m512 lcl_dbeta = _mm512_loadu_ps((__m512*)& beta_val_[v2] );

                __m512 lcl_dx_norm = _mm512_loadu_ps((__m512*)& dx_norm[v1 * n_feat + v2]);
                __m512 lcl_vstd_inv = _mm512_loadu_ps((__m512*)& std_inv[v2]);
                __m512 lcl_x_mu = _mm512_loadu_ps((__m512*)& x_mu[v1 * n_feat + v2]);
                
                __m512 lcl_dvar = _mm512_loadu_ps((__m512*)& dvar[v2]);
                __m512 lcl_dmu = _mm512_loadu_ps((__m512*)& dmu[v2]);
               
                __m512 lcl_dx_tmp1 = _mm512_mul_ps(lcl_dx_norm, lcl_vstd_inv);
                __m512 lcl_dx_tmp2 = _mm512_mul_ps(lcl_dvar, _mm512_mul_ps(lcl_vtwo, _mm512_mul_ps(lcl_x_mu, lcl_vrec_n)));
                __m512 lcl_dx_tmp3 = _mm512_mul_ps(lcl_dmu, lcl_vrec_n);
                __m512 lcl_dx = _mm512_add_ps(lcl_dx_tmp1, _mm512_add_ps(lcl_dx_tmp2, lcl_dx_tmp3));

                __m512 lcl_dgrad = _mm512_loadu_ps((__m512*)& dgrad[v1*n_feat + v2]);
                __m512 lcl_x_norm = _mm512_loadu_ps((__m512*)& inp_norm[v1*n_feat + v2]);

                lcl_dgamma = _mm512_add_ps(lcl_dgamma, _mm512_mul_ps(lcl_dgrad, lcl_x_norm));

                lcl_dbeta = _mm512_add_ps(lcl_dbeta, lcl_dgrad);

                _mm512_storeu_ps (( __m512*)&dinp_val[v1*n_feat + v2] , lcl_dx) ;
                _mm512_storeu_ps (( __m512*)&gamma_val[v2] , lcl_dgamma) ;
                _mm512_storeu_ps (( __m512*)&beta_val[v2] , lcl_dbeta) ;
            }

       if(v2 < n_feat)        
            {
                __m512 src;
                __mmask16 Msk = 0xFFFF  >> (16 - n_feat % 16); 

                __m512 lcl_dgamma = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& gamma_val_[v2] );
                __m512 lcl_dbeta = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& beta_val_[v2] );

                __m512 lcl_dx_norm = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dx_norm[v1 * n_feat + v2]);
                __m512 lcl_vstd_inv = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& std_inv[v2]);
                __m512 lcl_x_mu = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& x_mu[v1 * n_feat + v2]);
                
                __m512 lcl_dvar = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dvar[v2]);
                __m512 lcl_dmu = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dmu[v2]);
              
                __m512 lcl_dx_tmp1 = _mm512_mask_mul_ps(src,  Msk, lcl_dx_norm, lcl_vstd_inv);
                __m512 lcl_dx_tmp2 = _mm512_mask_mul_ps(src,  Msk, lcl_dvar, _mm512_mask_mul_ps(src,  Msk, lcl_vtwo, _mm512_mask_mul_ps(src,  Msk, lcl_x_mu, lcl_vrec_n)));
                __m512 lcl_dx_tmp3 = _mm512_mask_mul_ps(src,  Msk, lcl_dmu, lcl_vrec_n);
                __m512 lcl_dx = _mm512_mask_add_ps(src,  Msk, lcl_dx_tmp1, _mm512_mask_add_ps(src,  Msk, lcl_dx_tmp2, lcl_dx_tmp3));

                __m512 lcl_dgrad = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& dgrad[v1*n_feat + v2]);
                __m512 lcl_x_norm = _mm512_mask_loadu_ps(src,  Msk, (__m512*)& inp_norm[v1*n_feat + v2]);

                lcl_dgamma = _mm512_mask_add_ps(src,  Msk, lcl_dgamma, _mm512_mask_mul_ps(src,  Msk, lcl_dgrad, lcl_x_norm));

                lcl_dbeta = _mm512_mask_add_ps(src,  Msk, lcl_dbeta, lcl_dgrad);

                _mm512_mask_storeu_ps (( __m512*)&dinp_val[v1*n_feat + v2] , Msk, lcl_dx) ;
                _mm512_mask_storeu_ps (( __m512*)&gamma_val[v2] , Msk, lcl_dgamma) ;
                _mm512_mask_storeu_ps (( __m512*)&beta_val[v2] , Msk, lcl_dbeta) ;
            }

        }
    }
       
        for (int i = 1; i < (num_threads-1) * n_feat; ++i)
        {
			gamma_val_[0] += gamma_val_[i];
            beta_val_[0] += beta_val_[i];         	
        }
        gamma_fnl = &gamma_val_[0];
        beta_fnl = &beta_val_[0];

_mm_free(x_mu);
_mm_free(dx_norm);
_mm_free(std_inv);
_mm_free(neg_std_inv);
_mm_free(cube_std_inv);
_mm_free(dvar_sum_);
_mm_free(dxnorm_stdinv_sum_);
_mm_free(x_mu_sum_);
_mm_free(dvar);
_mm_free(dmu);
_mm_free(gamma_val_);
_mm_free(beta_val_);

                   
    return {grad_inp, grad_gamma, grad_beta};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batchnorm_fwd_impl", &batchnorm_fwd_impl, "batchnorm forwardpass c++ code");
  m.def("batchnorm_bwd_impl", &batchnorm_bwd_impl, "batchnorm backpass c++ code");  
}




