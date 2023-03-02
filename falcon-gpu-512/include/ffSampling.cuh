#include <stdio.h>
#include <stdint.h>
#include "fpr.cuh"
#include "cuda_kernel.cuh"
#include "rng.cuh"

// typedef struct
// {
// 	fpr *t0; 
// 	fpr *t1;
// 	fpr *g00; 
// 	fpr *g01; 
// 	fpr *g11;
// 	unsigned logn; 
// 	fpr *tmp;
// 	fpr *z0; 
// 	fpr *z1;
// } STACK;


typedef struct
 {
	fpr *t0; 
	fpr *g00; 
	fpr *g11;
	unsigned logn; 
	unsigned char is_z0, is_z1;
 } STACK; /* t1=t0+n, z0=tmp, z1=tmp+n, g01=g00+n, tmp=t0+2n */
	
typedef struct {
	prng_s p;
	fpr sigma_min;
} sampler_context_s;
// For GPU
typedef struct {
	prng_s p;
	fpr sigma_min;
} d_sampler_context_s;

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);
__global__ void ffSampling_fft_dyntree(fpr *t0, fpr *t1, fpr *g00, fpr *g01, fpr *g11,	unsigned orig_logn, unsigned logn, fpr *tmp, uint64_t *scA, uint64_t *scdptr);
