#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
// #include <mkl.h>
#include <iostream>
#include <cmath> 
#include <tuple>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>


template<typename T1, typename T2>
void my_copy(T1 *dst, T2 *src, int len) {
#pragma omp simd
  for(int64_t v = 0; v < len; v++) {
    dst[v] = src[v] ;
  }
}

template<typename T1, typename T2>
void my_add(T1 *inout, T2 *in, int len, float alpha) {
#pragma omp simd
  for(int64_t v = 0; v < len; v++) {
    inout[v] += in[v] * alpha;
  }
}

template<typename T>
void my_memcopy(T *dst, T *src, int len) {
    std::memcpy(dst, src, len*sizeof(T));
}

at::Tensor embedding_forward( at::Tensor& tweight, at::Tensor& tindices) 
{
    // std::cout << "Embedding Fwd" << std::endl;
    int nIndR = tindices.size(0);

   int nIndC = 1;

   int D = tweight.size(1);
   auto opts = tweight.options().dtype(at::kFloat);
   at::Tensor toutput;
   if (tindices.dim() == 1)
   {
        toutput = at::empty({nIndR, D}, opts);
    }
    else
    {
        nIndC = tindices.size(1);
        toutput = at::empty({nIndR, nIndC, D}, opts);
    }

   float* out_data = (float *) toutput.data_ptr();   
   if(tweight.is_contiguous() && tindices.is_contiguous()) 
    {
        float* w_data = (float *) tweight.data_ptr();
        int64_t* idx_data = tindices.data_ptr<int64_t>();
        #pragma omp parallel for
        for (int i = 0; i < nIndR*nIndC; ++i) 
        {
            int idx = idx_data[i]; 
            float* out_data_idx = &out_data[i*D];
            float* w_data_idx = &w_data[idx*D];
            my_copy(out_data_idx, w_data_idx, D); 
        }
    }
    else
    {
        printf("This path has some bug, need to debug... exiting!\n");
        exit(1);
    }
	return toutput;
}



at::Tensor embedding_backward( at::Tensor& tgrad, at::Tensor& tweight, at::Tensor& tindices) 
{
    int nIndR = tindices.size(0);
    int nIndC = 1; 
    int M = tweight.size(0);
    int D = tweight.size(1);

    auto tvalues = at::empty_like(tweight, tweight.options());
    float* values_data = tvalues.data_ptr<float>();
    memset(values_data, 0,  M * D * sizeof(float));

    float* grad_data = tgrad.data_ptr<float>();
    int64_t* idx_data = tindices.data_ptr<int64_t>(); 

    if (tindices.dim() == 1)
    {
        int nInd = nIndC*nIndR; 
        #pragma omp parallel for      
        for (int i = 0; i < nInd; ++i)
        {
            int64_t idx = idx_data[i];
            float* ga = &grad_data[i*D];
            float* va = &values_data[idx*D];
            my_add(va, ga, D, 1.0f);
        }
    }
    else
    {
        

        int nIndC = tindices.size(1);
        int nInd = nIndC*nIndR;

        #pragma omp parallel for
        for (int i = 0; i < nInd; ++i)
        {
            int64_t idx = idx_data[i];
            std::cout << "Idx: " << idx << std::endl;
            float* ga = &grad_data[i*D];
            float* va = &values_data[idx*D];
        #pragma omp critical
            my_add(va, ga, D, 1.0f);
        }      
    }

    return tvalues; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embedding_forward", &embedding_forward, "embedding forwardpass c++ code");
  m.def("embedding_backward", &embedding_backward, "embedding backpass c++ code");  
}
