#include <stdio.h>
#include "../include/fpr.cuh"
#include "../include/cuda_kernel.cuh"

// Include local CUDA header files.
// extern "C" {
// #include "include/cuda_kernel.cuh"
// }
// extern "C" void crypto_sign(double *h_c1, double *h_c2 /*, h_mr*/);
// extern void crypto_ver();


int main() {

    fpr *h_c1, *h_c2, *h_pk;
    uint8_t *h_mr;    

    cudaMallocHost((void**) &h_c1, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_c2, BATCH*N*sizeof(fpr)); 
    cudaMallocHost((void**) &h_mr, BATCH* (MSG_BYTES+MITAKA_K/8) * sizeof(uint8_t));
    cudaMallocHost((void**) &h_pk, BATCH*N*sizeof(fpr)); 
    crypto_sign(h_c1, h_c2, h_mr);
    crypto_ver(h_pk, h_c1, h_c2, h_mr);
    return 0;
}