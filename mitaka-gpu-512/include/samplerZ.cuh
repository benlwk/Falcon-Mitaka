#include <stdint.h>
#include "../include/fpr.cuh"
#include "../include/cuda_kernel.cuh"

#define TABLE_SIZE 13


__constant__ uint64_t CDT[TABLE_SIZE] = {8562458705743934607LLU,
                           14988938141546119862LLU,
                           17705984313312429518LLU,
                           18353082494776078532LLU,
                           18439897061947435901LLU,
                           18446457975170112665LLU,
                           18446737284374178633LLU,
                           18446743982533372247LLU,
                           18446744073018029834LLU,
                           18446744073706592852LLU,
                           18446744073709544480LLU,
                           18446744073709551607LLU,
                           18446744073709551615LLU};

__global__ void sample_discrete_gauss_gpu(fpr* p, uint8_t *prng);                           
__global__ void prng_refill_g(uint8_t *dst, uint8_t *prng_key);
__global__ void prng_refill_SM_g(uint8_t *dst, uint8_t *prng_key);
