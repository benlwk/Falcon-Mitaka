
/*
 * SHAKE implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2017-2019  Falcon Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@nccgroup.com>
 */

#include <string.h>
#include "../include/shake.cuh"
#include "../include/fft.cuh"

/*************************************************
 * Name:        load64
 *
 * Description: Load 8 bytes into uint64_t in little-endian order
 *
 * Arguments:   - const uint8_t *x: pointer to input byte array
 *
 * Returns the loaded 64-bit unsigned integer
 **************************************************/
__device__ static uint64_t load64(const uint8_t *x) {
    uint64_t r = 0;
    for (size_t i = 0; i < 8; ++i) {
        r |= (uint64_t)x[i] << 8 * i;
    }

    return r;
}

/*************************************************
 * Name:        store64
 *
 * Description: Store a 64-bit integer to a byte array in little-endian order
 *
 * Arguments:   - uint8_t *x: pointer to the output byte array
 *              - uint64_t u: input 64-bit unsigned integer
 **************************************************/
__device__ static void store64(uint8_t *x, uint64_t u) {
    for (size_t i = 0; i < 8; ++i) {
        x[i] = (uint8_t) (u >> 8 * i);
    }
}


__global__ void shake128_absorb_gpu(uint64_t *out, uint8_t *in, uint64_t inlen) 
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint8_t p = 0x1F;   // For absorb
    uint32_t r = 168;   // For shake128
    uint32_t s = threadIdx.x % 5, i;
    __shared__ uint8_t t[200];

    // Absorb phase       
	// This is less than one block, no Keccak is applied.
    // wklee, only load MSG_BYTES+MITAKA_K/8 bytes (9*8)
    for(i=0; i<2; i++) t[i*25 + tid] = in[bid*(MSG_BYTES+MITAKA_K/8) + i*25 + tid];// 50b
    if(tid<22) t[i*25 + tid] = in[bid*(MSG_BYTES+MITAKA_K/8) + i*25 + tid];	// 22b

    if(tid==0) 
    {
        t[inlen] = p;
        t[r - 1] |= 128;
    }
    __syncthreads();     
    out[bid*25 +tid] ^= load64(t + 8*tid);    
}

__global__ void shake128_absorb_gpu2(uint64_t *out, uint8_t *in, uint64_t inlen) 
{
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint8_t p = 0x1F;   // For absorb
    uint32_t r = 168;   // For shake128
    uint32_t s = threadIdx.x % 5, i;
    __shared__ uint8_t t[200];
    for(i=0; i<200/blockDim.x; i++) t[i*blockDim.x + tid] = 0;
    // Absorb phase       
    // This is less than one block, no Keccak is applied.
    // wklee, only load MSG_BYTES+MITAKA_K/8 bytes (9*8)
    for(i=0; i<2; i++) t[i*25 + tid] = in[bid*(MSG_BYTES+MITAKA_K/8) + i*25 + tid];// 50b
    if(tid<22) t[i*25 + tid] = in[bid*(MSG_BYTES+MITAKA_K/8) + i*25 + tid]; // 22b

    if(tid==0) 
    {
        t[inlen] = p;
        t[r - 1] |= 128;
    }
    __syncthreads();     
    out[bid*25 +tid] ^= load64(t + 8*tid);    
    // out[bid*25 +tid] = 0;
}

// wklee, fine-grain version of SHAKE, 25 threads per SHAKE
// __global__ void keccak_gpu(uint64_t *in) 
__device__ void keccak_gpu(uint64_t *A) 
{
    uint32_t tid = threadIdx.x;
    uint32_t s = threadIdx.x % 5;
    
    __shared__ uint64_t C[25];
    __shared__ uint64_t D[25];

    if (tid < 25) {                      
        for (int i = 0; i<NROUNDS; ++i) {
            C[tid] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
            D[tid] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
            C[tid] = R64(A[a[tid]] ^ D[b[tid]], ro[tid][0], ro[tid][1]);
            A[d[tid]] = C[c[tid][0]] ^ ((~C[c[tid][1]]) & C[c[tid][2]]);            
            A[tid] ^= rc[(tid == 0) ? 0 : 1][i];
        }          
    }       
}

__global__ void shake128_squeezeblocks(fpr *out, uint64_t *state)
{
    __shared__ uint8_t buf[SHAKE128_RATE];
    uint32_t tid = threadIdx.x, bid = blockIdx.x, i, j;
    uint32_t ctr = 0;
    uint16_t val;

    while(ctr < MITAKA_D) 
    {
        keccak_gpu(state + bid*25);    
        if(tid<21)store64(buf + 8*tid, state[bid*25 + tid]);
        for(i=0;i<SHAKE128_RATE && ctr < MITAKA_D;i+=2)
        {
            val = (buf[i] | ((uint16_t) buf[i+1] << 8));
            if(val < 5*MITAKA_Q)
            {
                out[bid*N + ctr].v = (double)(val%MITAKA_Q);
                ctr++;
            }
        }
    }
}

__global__ void shake128_squeezeblocks_u(uint32_t *out, uint64_t *state)
{
    __shared__ uint8_t buf[SHAKE128_RATE];
    uint32_t tid = threadIdx.x, bid = blockIdx.x, i, j;
    uint32_t ctr = 0;
    uint16_t val;

    while(ctr < MITAKA_D) 
    {
        keccak_gpu(state + bid*25);    
        if(tid<21)store64(buf + 8*tid, state[bid*25 + tid]);
        for(i=0;i<SHAKE128_RATE && ctr < MITAKA_D;i+=2)
        {
            val = (buf[i] | ((uint16_t) buf[i+1] << 8));
            if(val < 5*MITAKA_Q)
            {
                out[bid*N + ctr] = (val%MITAKA_Q);
                ctr++;
            }
        }
    }
}

// wklee, fine grain version, 25 threads.
__global__ void hash_to_point_vartime_par(uint64_t *g_scA, uint64_t *scdptr, uint16_t *x)
{
	uint64_t dptr;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	uint32_t len, countN;	
	uint8_t out[2];
	uint32_t w, i, n=N;
	__shared__ uint64_t scA[25];

	dptr = scdptr[bid];
	scA[tid] = g_scA[bid*25 + tid];
	__syncthreads();
	countN = 0;
	
	while(n > 0){
		len = 2;	// wklee, always processes 2 bytes only
		while (len > 0) {
			size_t clen;

			if (dptr == 136) {
				keccak_gpu(scA);
				// keccak_gpu(scA + bid*25);
				// keccak_gpu_warp(scA + bid*25);
				dptr = 0;
			}
			__syncthreads();
			len -= 2;
			clen = 2;
			
			out[0] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
			dptr ++;
			out[1] = scA[(dptr >> 3)] >> ((dptr & 7) << 3);
			dptr ++;
		}
		__syncthreads();
		
		w = ((unsigned)out[0] << 8) | (unsigned)out[1];
		if (w < 61445) {
			while (w >= 12289) {
				w -= 12289;
			}				
			x[bid*N + countN] = (uint16_t)w;
			n--;		
			countN++;		
		}
	}
	__syncthreads();
	
	scdptr[bid] = dptr;
}

__global__ void normaldist_g(fpr *vec, uint64_t *u, uint64_t *v, uint64_t *e)
{
    int i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    __shared__ double uf[MITAKA_D/2], vf[MITAKA_D/2];
    __shared__ int geom[MITAKA_D/2];
    double uf_, vf_;

    uf_ = 2*M_PI*(double)(u[bid*N/2 + tid] & 0x1FFFFFFFFFFFFFul) * pow(2,-53);
    vf_ = 0.5 + (double)(v[bid*N/2 +tid] & 0x1FFFFFFFFFFFFFul) * pow(2,-54);

    geom[tid] = CMUX(63 + __ffsll(e[bid*N + 2*tid+1]), __ffsll(e[bid*N + 2*tid]) - 1, 
        CZERO64(e[bid*N + 2*tid]));

    vf_ = sqrt(MITAKA_D*(M_LN2*geom[tid]-log(vf_)));
    // __syncthreads();
    
    vec[bid*N + 2*tid].v   = vf_ * cos(uf_);
    vec[bid*N + 2*tid+1].v = vf_ * sin(uf_);

}

__global__ void normaldist_mul_fft_g(fpr *g_vec, uint64_t *u, uint64_t *v, uint64_t *e, fpr *b)
{
    int i;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    __shared__ double uf[MITAKA_D/2], vf[MITAKA_D/2];
    __shared__ fpr vec[MITAKA_D];
    __shared__ int geom[MITAKA_D/2];

    uf[tid] = 2*M_PI*(double)(u[bid*N/2 + tid] & 0x1FFFFFFFFFFFFFul) * pow(2,-53);
    vf[tid] = 0.5 + (double)(v[bid*N/2 +tid] & 0x1FFFFFFFFFFFFFul) * pow(2,-54);

    geom[tid] = CMUX(63 + __ffsll(e[bid*N + 2*tid+1]), __ffsll(e[bid*N + 2*tid]) - 1, 
        CZERO64(e[bid*N + 2*tid]));

    vf[tid] = sqrt(MITAKA_D*(M_LN2*geom[tid]-log(vf[tid])));
    
    vec[2*tid].v   = vf[tid] * cos(uf[tid]);
    vec[2*tid+1].v = vf[tid] * sin(uf[tid]);
    // __syncthreads();

    // FFT point-wise multiplication
    uint32_t hn;
        
    hn = N >> 1;
    fpr a_re, a_im, b_re, b_im;

    // a_re = vec[bid*N + tid];    
    // a_im = vec[bid*N + tid + hn];
    // b_re = b[bid*N + tid];
    // b_im = b[bid*N + tid + hn];    
    a_re = vec[tid];    
    a_im = vec[tid + hn];
    b_re = b[tid];
    b_im = b[tid + hn];    
    
    FPC_MUL(g_vec[bid*N + tid], g_vec[bid*N + tid + hn], a_re, a_im, b_re, b_im);    

}
