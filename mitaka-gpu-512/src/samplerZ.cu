#include "../include/samplerZ.cuh"
#include <stdio.h>

__device__ static uint64_t load64(uint8_t *x) {
    uint64_t r = 0;
    for (size_t i = 0; i < 8; ++i) {
        r |= (uint64_t)x[i] << 8 * i;
    }

    return r;
}

__device__ static uint32_t load32(uint8_t *x) {
    uint32_t r = 0;
    for (size_t i = 0; i < 4; ++i) {
        r |= (uint32_t)x[i] << 8 * i;
    }

    return r;
}

/*
 * PRNG based on ChaCha20.
 *
 * State consists in key (32 bytes) then IV (16 bytes) and block counter
 * (8 bytes). 
 */
__constant__ uint32_t CW[] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

#define QROUND(a, b, c, d)   do { \
        state[a] += state[b]; \
        state[d] ^= state[a]; \
        state[d] = (state[d] << 16) | (state[d] >> 16); \
        state[c] += state[d]; \
        state[b] ^= state[c]; \
        state[b] = (state[b] << 12) | (state[b] >> 20); \
        state[a] += state[b]; \
        state[d] ^= state[a]; \
        state[d] = (state[d] <<  8) | (state[d] >> 24); \
        state[c] += state[d]; \
        state[b] ^= state[c]; \
        state[b] = (state[b] <<  7) | (state[b] >> 25); \
    } while (0)

#define QROUNDSM(a, b, c, d)   do { \
        state[tid*16 + a] += state[tid*16 + b]; \
        state[tid*16 + d] ^= state[tid*16 + a]; \
        state[tid*16 + d] = (state[tid*16 + d] << 16) | (state[tid*16 + d] >> 16); \
        state[tid*16 + c] += state[tid*16 + d]; \
        state[tid*16 + b] ^= state[tid*16 + c]; \
        state[tid*16 + b] = (state[tid*16 + b] << 12) | (state[tid*16 + b] >> 20); \
        state[tid*16 + a] += state[tid*16 + b]; \
        state[tid*16 + d] ^= state[tid*16 + a]; \
        state[tid*16 + d] = (state[tid*16 + d] <<  8) | (state[tid*16 + d] >> 24); \
        state[tid*16 + c] += state[tid*16 + d]; \
        state[tid*16 + b] ^= state[tid*16 + c]; \
        state[tid*16 + b] = (state[tid*16 + b] <<  7) | (state[tid*16 + b] >> 25); \
    } while (0)
/*
 * PRNG based on ChaCha20.
 *
 * State consists in key (32 bytes) then IV (16 bytes) and block counter
 * (8 bytes). Normally, we should not care about local endianness (this
 * is for a PRNG), but for the NIST competition we need reproducible KAT
 * vectors that work across architectures, so we enforce little-endian
 * interpretation where applicable. Moreover, output words are "spread
 * out" over the output buffer with the interleaving pattern that is
 * naturally obtained from the AVX2 implementation that runs eight
 * ChaCha20 instances in parallel.
 *
 * The block counter is XORed into the first 8 bytes of the IV.
 */
__global__ void prng_refill_g(uint8_t *dst, uint8_t *prng_key)
{    
    uint64_t u, i, j;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    uint64_t cc;
    uint8_t buf_d[NSAMP];
    uint32_t state[16*N];

    for (j = 0; j < NSAMP/64; j ++) {                
        // cc = *(uint64_t *)(state_d + 48);    
        cc = bid*N*NSAMP + tid * NSAMP + j*8;
        
        for (u = 0; u < 8; u ++) {
            
            uint64_t v;        

            // memcpy(&state[0], CW, sizeof CW);
            // memcpy(&state[4], prng_key+bid*56, 48);            
            for (i = 0; i < 4; ++i) state[i] = CW[i];
            for (i = 0; i < 12; ++i) state[4 + i] = load32(prng_key+bid*56+4*i);
            state[14] ^= (uint32_t)cc;
            state[15] ^= (uint32_t)(cc >> 32);
            // printf("%lu %u\n", u, state[15]);
            for (i = 0; i < 10; i ++) {
                QROUND( 0,  4,  8, 12);
                QROUND( 1,  5,  9, 13);
                QROUND( 2,  6, 10, 14);
                QROUND( 3,  7, 11, 15);
                QROUND( 0,  5, 10, 15);
                QROUND( 1,  6, 11, 12);
                QROUND( 2,  7,  8, 13);
                QROUND( 3,  4,  9, 14);
            }

            for (v = 0; v < 4; v ++) {
                state[v] += CW[v];
            }
            for (v = 4; v < 14; v ++) {
                state[v] += ((uint32_t *)prng_key)[bid*14 + v - 4];
            }
            state[14] += ((uint32_t *)prng_key)[bid*14 +10]
                ^ (uint32_t)cc;
            state[15] += ((uint32_t *)prng_key)[bid*14 +11]
                ^ (uint32_t)(cc >> 32);
            // printf("%u %lu %u\n", j, u, state[15]);
            cc ++;

            /*
             * We mimic the interleaving that is used in the AVX2 implementation.
             */
            for (v = 0; v < 16; v ++) {
                buf_d[(u << 2) + (v << 5) + 0] =
                    (uint8_t)state[v];
                buf_d[(u << 2) + (v << 5) + 1] =
                    (uint8_t)(state[v] >> 8);
                buf_d[(u << 2) + (v << 5) + 2] =
                    (uint8_t)(state[v] >> 16);
                buf_d[(u << 2) + (v << 5) + 3] =
                    (uint8_t)(state[v] >> 24);
            }
        }
        for (i = 0; i < 64; ++i)
            // dst[bid*N*NSAMP + tid*NSAMP + j*64 + i] = buf_d[i];
          dst[bid*N*NSAMP + tid + j*N*64 + i*N] = buf_d[i];
    }
}

// __global__ void prng_refill_SM_g(uint8_t *dst, uint8_t *prng_key)
// {    
//     uint64_t u, i, j;
//     uint32_t tid = threadIdx.x, bid = blockIdx.x;
//     uint64_t cc;
//     uint8_t buf_d[NSAMP];
//     __shared__ uint32_t state[16*N];

//     for (j = 0; j < NSAMP/64; j ++) {                
//         // cc = *(uint64_t *)(state_d + 48);    
//         cc = bid*N*NSAMP + tid * NSAMP + j*8;
        
//         for (u = 0; u < 8; u ++) {
//             uint64_t v;               
//             for (i = 0; i < 4; ++i) state[tid*16 + i] = CW[i];
//             for (i = 0; i < 48; ++i) state[tid*16 +4 + i] = load32(prng_key+bid*56+4*i);
//             state[tid*16 +14] ^= (uint32_t)cc;
//             state[tid*16 +15] ^= (uint32_t)(cc >> 32);
            
//             for (i = 0; i < 10; i ++) {
//                 QROUNDSM( 0,  4,  8, 12);
//                 QROUNDSM( 1,  5,  9, 13);
//                 QROUNDSM( 2,  6, 10, 14);
//                 QROUNDSM( 3,  7, 11, 15);
//                 QROUNDSM( 0,  5, 10, 15);
//                 QROUNDSM( 1,  6, 11, 12);
//                 QROUNDSM( 2,  7,  8, 13);
//                 QROUNDSM( 3,  4,  9, 14);
//             }

//             for (v = 0; v < 4; v ++) {
//                 state[tid*16 +v] += CW[v];
//             }
//             for (v = 4; v < 14; v ++) {
//                 state[tid*16 +v] += ((uint32_t *)prng_key)[bid*14 + v - 4];
//             }
//             state[tid*16 +14] += ((uint32_t *)prng_key)[bid*14 +10]
//                 ^ (uint32_t)cc;
//             state[tid*16 +15] += ((uint32_t *)prng_key)[bid*14 +11]
//                 ^ (uint32_t)(cc >> 32);            
//             cc ++;

//             /*
//              * We mimic the interleaving that is used in the AVX2 implementation.
//              */
//             for (v = 0; v < 16; v ++) {
//                 buf_d[(u << 2) + (v << 5) + 0] =
//                     (uint8_t)state[tid*16 +v];
//                 buf_d[(u << 2) + (v << 5) + 1] =
//                     (uint8_t)(state[tid*16 +v] >> 8);
//                 buf_d[(u << 2) + (v << 5) + 2] =
//                     (uint8_t)(state[tid*16 +v] >> 16);
//                 buf_d[(u << 2) + (v << 5) + 3] =
//                     (uint8_t)(state[tid*16 +v] >> 24);
//             }
//         }
//         for (i = 0; i < 64; ++i)
//             // dst[bid*N*NSAMP + tid*NSAMP + j*64 + i] = buf_d[i];
//           dst[bid*N*NSAMP + tid + j*N*64 + i*N] = buf_d[i];
//     }
// }

__device__ int base_sampler(uint8_t *prng, uint64_t *CDT_SM){
  // uint64_t r = get64();
  uint64_t r = load64(prng);
  int res=0;
  for(int i=0; i < TABLE_SIZE; ++i)
    res += (r >= CDT_SM[i]);
  return res;
}

__device__ int samplerZ(double u, uint8_t *prng){
  int z0, b, z, uf;
  double x, p, r;
  uint8_t entropy;
  uf = floor(u);
  uint32_t tid = threadIdx.x, count=0;  
  __shared__ uint64_t CDT_SM[TABLE_SIZE];

  if(tid<TABLE_SIZE) CDT_SM[tid] = CDT[tid];
  __syncthreads();
  
  while (1){    
    // entropy = get8();
    entropy = prng[0 + tid*NSAMP + count*N*NSAMP/2];
    for(int i=0; i < 8; ++i){
      z0 = base_sampler(prng + tid*NSAMP + count*N*NSAMP/2 + i*16, CDT_SM);
      b = (entropy >> i)&1;
      z = (2*b-1)*z0 + b + uf; // add floor u here because we subtract u at the next step
      x = ((double)(z0*z0)-((double)(z-u)*(z-u)))/(2*R*R);
      p = exp(x);
      
      /*
      r = (double)get64()/(1LLU<<63);
      r /= 2;
      */
      r = (double)(load64(prng + tid*NSAMP + count*N*NSAMP/2 + i*16 + 8) & 0x1FFFFFFFFFFFFFul) * pow(2,-53);
      // r = (double)(1 & 0x1FFFFFFFFFFFFFul) * pow(2,-53);
      if (r < p)
        return z;
      count++;
    }
  }

}


__global__ void sample_discrete_gauss_gpu(fpr* p, uint8_t *prng){

    uint32_t tid = threadIdx.x, bid = blockIdx.x;  
    p[bid*N + tid].v = samplerZ(p[bid*N + tid].v, prng + bid*N*NSAMP);
  
}