#include "../include/fft.cuh"
#include "../include/consts.cuh"


// wklee, below are the parallel versions on GPU
__global__ void poly_set_g(fpr *out, const uint16_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    out[bid*10*N + tid] = fpr_of(in[bid*N + tid]);   
}


__global__ void poly_mulselfadj_fft_g(fpr *a)
{
    /*
     * Since each coefficient is multiplied with its own conjugate,
     * the result contains only real values.
     */
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    
    hn = N >> 1;    
    fpr a_re, a_im;
    a_re = a[bid*10*N + tid];
    a_im = a[bid*10*N + tid + hn];
    a[bid*10*N + tid] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    a[bid*10*N + tid + hn] = fpr_zero;
    
}

__global__ void poly_muladj_fft_g(fpr *a, const fpr *b)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    hn = N >> 1;

    fpr a_re, a_im, b_re, b_im;

    a_re = a[bid*10*N + tid];
    a_im = a[bid*10*N + tid + hn];
    b_re = b[bid*10*N + tid];
    b_im = fpr_neg(b[bid*10*N + tid + hn]);
    FPC_MUL(a[bid*10*N + tid], a[bid*10*N + tid + hn], a_re, a_im, b_re, b_im);
}

__global__ void poly_add_g(fpr *a, const fpr *b)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    a[bid*10*N + tid] = fpr_add(a[bid*10*N + tid], b[bid*10*N + tid]);    
}


//wklee, shared memory
__global__ void FFT_SM_g(fpr *f, uint32_t in_s, uint32_t out_s)
{
    uint32_t u, tid = threadIdx.x, bid = blockIdx.x;
    uint32_t t, hn, m;
    uint32_t j, j2;
    fpr x_re, x_im, y_re, y_im, s_re, s_im;      
    uint32_t ht, i1=0, j1=0;
    __shared__ fpr s_f[N];
    // __shared__ fpr s_fpr_gm_tab[2*N];

    hn = N >> 1;
    t = hn;
    m = 2;
    for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f[bid*in_s + u*blockDim.x + tid].v;
    // __syncthreads();

    // for (u = 0;  u < 2*N/blockDim.x; u ++)    
    //     s_fpr_gm_tab[u*blockDim.x + tid].v = fpr_gm_tab[u*blockDim.x + tid].v;
    __syncthreads();
// #pragma unroll 9    
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = s_f[j];
        x_im = s_f[j + hn];
        y_re = s_f[j + ht];
        y_im = s_f[j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(s_f[j], s_f[j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(s_f[j + ht], s_f[j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }
    for (u = 0;  u < N/blockDim.x; u ++)    
        f[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;
}

//wklee, shared memory
__global__ void FFT_SMx4_g(fpr *f0, fpr *f1, fpr *f2, fpr *f3, uint32_t in_s, uint32_t out_s)
{
    uint32_t u, tid = threadIdx.x, bid = blockIdx.x;
    uint32_t t, hn, m;
    uint32_t j, j2;
    fpr x_re, x_im, y_re, y_im, s_re, s_im;      
    uint32_t ht, i1=0, j1=0;
    __shared__ fpr s_f[N];

    hn = N >> 1;
    t = hn;
    m = 2;
    for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f0[bid*in_s + u*blockDim.x + tid].v;
    __syncthreads();
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = s_f[j];
        x_im = s_f[j + hn];
        y_re = s_f[j + ht];
        y_im = s_f[j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(s_f[j], s_f[j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(s_f[j + ht], s_f[j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }
    for (u = 0;  u < N/blockDim.x; u ++)    
        f0[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;
    // __syncthreads();
    hn = N >> 1;
    t = hn;    m = 2; 
    i1=0; j1=0;
    // second vector
	for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f1[bid*in_s + u*blockDim.x + tid].v;
    __syncthreads();
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = s_f[j];
        x_im = s_f[j + hn];
        y_re = s_f[j + ht];
        y_im = s_f[j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(s_f[j], s_f[j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(s_f[j + ht], s_f[j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }
    for (u = 0;  u < N/blockDim.x; u ++)    
        f1[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;   
    hn = N >> 1;
    t = hn;    m = 2; 
    i1=0; j1=0;
    // third vector
	for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f2[bid*in_s + u*blockDim.x + tid].v;
    __syncthreads();
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = s_f[j];
        x_im = s_f[j + hn];
        y_re = s_f[j + ht];
        y_im = s_f[j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(s_f[j], s_f[j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(s_f[j + ht], s_f[j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }
    for (u = 0;  u < N/blockDim.x; u ++)    
        f2[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;        
    hn = N >> 1;
    t = hn;    m = 2; 
    i1=0; j1=0;
    // fourth vector
	for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f3[bid*in_s + u*blockDim.x + tid].v;
    __syncthreads();
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = s_f[j];
        x_im = s_f[j + hn];
        y_re = s_f[j + ht];
        y_im = s_f[j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(s_f[j], s_f[j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(s_f[j + ht], s_f[j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }
    for (u = 0;  u < N/blockDim.x; u ++)    
        f3[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;               
}



// wklee, basic version
__global__ void FFT_g(fpr *f, uint32_t in_s, uint32_t out_s)
{
    uint32_t u, tid = threadIdx.x, bid = blockIdx.x;
    uint32_t t, hn, m;
    uint32_t j, j2;
    fpr x_re, x_im, y_re, y_im, s_re, s_im;      
    uint32_t ht, i1=0, j1=0;
    __shared__ fpr s_fpr_gm_tab[2*N];

    hn = N >> 1;
    t = hn;
    m = 2;
    //wklee, SM is slower, why?
    for (u = 0;  u < 2*N/blockDim.x; u ++)    
        s_fpr_gm_tab[u*blockDim.x + tid].v = fpr_gm_tab[u*blockDim.x + tid].v;
    __syncthreads();
    for (u = 1;  u < 9; u ++) {
        i1=0, j1=0;
        
        ht = t >> 1;        
            
        j2 = j1 + ht;
        i1 = (tid/j2);
        j = tid%j2 + (tid/j2)*2*j2;
        
        s_re = s_fpr_gm_tab[((m + i1) << 1) + 0];
        s_im = s_fpr_gm_tab[((m + i1) << 1) + 1];           

        x_re = f[bid*in_s + j];
        x_im = f[bid*in_s + j + hn];
        y_re = f[bid*in_s + j + ht];
        y_im = f[bid*in_s + j + ht + hn];

        FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
        FPC_ADD(f[bid*out_s + j], f[bid*out_s + j + hn],
            x_re, x_im, y_re, y_im);
        FPC_SUB(f[bid*out_s + j + ht], f[bid*out_s + j + ht + hn],
            x_re, x_im, y_re, y_im);
        j1 += t;

        m <<= 1;
        t = ht;
        __syncthreads();
    }

}

__global__ void iFFT_g(fpr *f, uint32_t in_s, uint32_t out_s)
{
	uint32_t hn, t, m, j;
	uint32_t u, tid = threadIdx.x, bid = blockIdx.x;	
	t = 1;
	m = N;
	hn = N >> 1;
	for (u = 9; u > 1; u --) {
		uint32_t hm, dt;
		hm = m >> 1;
		dt = t << 1;
		j = tid%t + (tid/t)*dt;
		fpr s_re, s_im;

		s_re = fpr_gm_tab[(hm*2 + (tid/t)*2) + 0];
		s_im = fpr_neg(fpr_gm_tab[(hm*2 + (tid/t)*2) + 1]);

		fpr x_re, x_im, y_re, y_im;
		x_re = f[bid*in_s + j];
		x_im = f[bid*in_s + j + hn];
		y_re = f[bid*in_s + j + t];
		y_im = f[bid*in_s + j + t + hn];
		FPC_ADD(f[bid*out_s + j], f[bid*out_s + j + hn],
			x_re, x_im, y_re, y_im);
		FPC_SUB(x_re, x_im, x_re, x_im, y_re, y_im);
		FPC_MUL(f[bid*out_s + j + t], f[bid*out_s + j + t + hn],
			x_re, x_im, s_re, s_im);

		__syncthreads();
		t = dt;
		m = hm;
	}

	for (u = 0; u < N/blockDim.x; u ++) {
		f[bid*out_s + u*blockDim.x + tid] = fpr_mul(f[bid*out_s + u*blockDim.x + tid], fpr_p2_tab[9]);
	}
}

__global__ void poly_mul_fft(fpr *a, const fpr *b)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    
    hn = N >> 1;
    fpr a_re, a_im, b_re, b_im;

    a_re = a[bid*10*N + tid];
    a_im = a[bid*10*N + tid + hn];
    b_re = b[bid*10*N + tid];
    b_im = b[bid*10*N + tid + hn];
    FPC_MUL(a[bid*10*N + tid], a[bid*10*N + tid + hn], a_re, a_im, b_re, b_im);
}

__global__ void poly_mulconst(fpr *a, fpr x)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    a[bid*10*N + tid] = fpr_mul(a[bid*10*N + tid], x);
}

__global__ void smallints_to_fpr_g(fpr *r, int8_t *t, uint32_t in_s, uint32_t out_s)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    r[bid*out_s + tid].v = (double)t[bid*in_s + tid];    
}

__global__ void poly_neg_g(fpr *a, uint32_t in_s, uint32_t out_s)
{    
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    a[bid*out_s + tid] = fpr_neg(a[bid*in_s + tid]);
}

__global__ void poly_copy(fpr *out, fpr *in)
{    
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    out[bid*10*N + tid] = in[bid*10*N + tid];
}

__global__ void byte_copy(uint8_t *out, uint8_t *in, uint32_t outlen, uint32_t inlen)
{    
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    out[bid*outlen + tid] = in[bid*inlen + tid];
}

__global__ void byte_copy2(uint8_t *out, uint8_t *in, uint32_t outlen, uint32_t *inlen)
{    
    uint32_t bid = blockIdx.x, i;

    for(i=0; i<inlen[bid]; i++)
    	out[bid*outlen + i] = in[bid*(CRYPTO_BYTES - 2 - NONCELEN) + i];
}

// wklee, serial implementation
__device__ void poly_LDL_fft(const fpr * g00,
	fpr * g01, fpr * g11, unsigned logn)
{
	size_t n, hn, u;

	n = (size_t)1 << logn;
	hn = n >> 1;
	for (u = 0; u < hn; u ++) {
		fpr g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
		fpr mu_re, mu_im;

		g00_re = g00[u];
		g00_im = g00[u + hn];
		g01_re = g01[u];
		g01_im = g01[u + hn];
		g11_re = g11[u];
		g11_im = g11[u + hn];
		FPC_DIV(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
		FPC_MUL(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg(g01_im));
		FPC_SUB(g11[u], g11[u + hn], g11_re, g11_im, g01_re, g01_im);
		g01[u] = mu_re;
		g01[u + hn] = fpr_neg(mu_im);
	}
}

__device__ void poly_split_fft(fpr * f0, fpr * f1,
	const fpr * f, unsigned logn)
{
	/*
	 * The FFT representation we use is in bit-reversed order
	 * (element i contains f(w^(rev(i))), where rev() is the
	 * bit-reversal function over the ring degree. This changes
	 * indexes with regards to the Falcon specification.
	 */
	size_t n, hn, qn, u;

	n = (size_t)1 << logn;
	hn = n >> 1;
	qn = hn >> 1;

	/*
	 * We process complex values by pairs. For logn = 1, there is only
	 * one complex value (the other one is the implicit conjugate),
	 * so we add the two lines below because the loop will be
	 * skipped.
	 */
	f0[0] = f[0];
	f1[0] = f[hn];

	for (u = 0; u < qn; u ++) {
		fpr a_re, a_im, b_re, b_im;
		fpr t_re, t_im;

		a_re = f[(u << 1) + 0];
		a_im = f[(u << 1) + 0 + hn];
		b_re = f[(u << 1) + 1];
		b_im = f[(u << 1) + 1 + hn];

		FPC_ADD(t_re, t_im, a_re, a_im, b_re, b_im);
		f0[u] = fpr_half(t_re);
		f0[u + qn] = fpr_half(t_im);

		FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
		FPC_MUL(t_re, t_im, t_re, t_im,
			fpr_gm_tab[((u + hn) << 1) + 0],
			fpr_neg(fpr_gm_tab[((u + hn) << 1) + 1]));
		f1[u] = fpr_half(t_re);
		f1[u + qn] = fpr_half(t_im);
	}
}

__device__ void poly_merge_fft(fpr * f,
	const fpr * f0, const fpr * f1, unsigned logn)
{
	size_t n, hn, qn, u;

	n = (size_t)1 << logn;
	hn = n >> 1;
	qn = hn >> 1;

	/*
	 * An extra copy to handle the special case logn = 1.
	 */
	f[0] = f0[0];
	f[hn] = f1[0];

	for (u = 0; u < qn; u ++) {
		fpr a_re, a_im, b_re, b_im;
		fpr t_re, t_im;

		a_re = f0[u];
		a_im = f0[u + qn];
		FPC_MUL(b_re, b_im, f1[u], f1[u + qn],
			fpr_gm_tab[((u + hn) << 1) + 0],
			fpr_gm_tab[((u + hn) << 1) + 1]);
		FPC_ADD(t_re, t_im, a_re, a_im, b_re, b_im);
		f[(u << 1) + 0] = t_re;
		f[(u << 1) + 0 + hn] = t_im;
		FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
		f[(u << 1) + 1] = t_re;
		f[(u << 1) + 1 + hn] = t_im;
	}
}

// void poly_mul_fft_s(fpr *a, const fpr *b, unsigned logn)
// {
// 	size_t n, hn, u;

// 	n = (size_t)1 << logn;
// 	hn = n >> 1;
// 	for (u = 0; u < hn; u ++) {
// 		fpr a_re, a_im, b_re, b_im;

// 		a_re = a[u];
// 		a_im = a[u + hn];
// 		b_re = b[u];
// 		b_im = b[u + hn];
// 		FPC_MUL2(a[u], a[u + hn], a_re, a_im, b_re, b_im);
// 	}
// }
__device__ void poly_add(
	fpr * a, const fpr * b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_add(a[u], b[u]);
	}
}

__device__ void poly_sub(
	fpr * a, const fpr * b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_sub(a[u], b[u]);
	}
}


/*
 * Addition modulo q. Operands must be in the 0..q-1 range.
 */
__device__ static inline uint32_t mq_add(uint32_t x, uint32_t y)
{
	/*
	 * We compute x + y - q. If the result is negative, then the
	 * high bit will be set, and 'd >> 31' will be equal to 1;
	 * thus '-(d >> 31)' will be an all-one pattern. Otherwise,
	 * it will be an all-zero pattern. In other words, this
	 * implements a conditional addition of q.
	 */
	uint32_t d;

	d = x + y - Q;
	d += Q & -(d >> 31);
	return d;
}

/*
 * Subtraction modulo q. Operands must be in the 0..q-1 range.
 */
__device__ static inline uint32_t mq_sub(uint32_t x, uint32_t y)
{
	/*
	 * As in mq_add(), we use a conditional addition to ensure the
	 * result is in the 0..q-1 range.
	 */
	uint32_t d;

	d = x - y;
	d += Q & -(d >> 31);
	return d;
}

/*
 * Division by 2 modulo q. Operand must be in the 0..q-1 range.
 */
// __device__ static inline uint32_t mq_rshift1(uint32_t x)
// {
// 	x += Q & -(x & 1);
// 	return (x >> 1);
// }

/*
 * Montgomery multiplication modulo q. If we set R = 2^16 mod q, then this function computes: x * y / R mod q
 * Operands must be in the 0..q-1 range.
 */
__device__ static inline uint32_t mq_montymul(uint32_t x, uint32_t y)
{
	uint32_t z, w;

	/*
	 * We compute x*y + k*q with a value of k chosen so that the 16
	 * low bits of the result are 0. We can then shift the value.
	 * After the shift, result may still be larger than q, but it
	 * will be lower than 2*q, so a conditional subtraction works.
	 */

	z = x * y;
	w = ((z * Q0I) & 0xFFFF) * Q;

	/*
	 * When adding z and w, the result will have its low 16 bits
	 * equal to 0. Since x, y and z are lower than q, the sum will
	 * be no more than (2^15 - 1) * q + (q - 1)^2, which will
	 * fit on 29 bits.
	 */
	z = (z + w) >> 16;

	/*
	 * After the shift, analysis shows that the value will be less
	 * than 2q. We do a subtraction then conditional subtraction to
	 * ensure the result is in the expected range.
	 */
	z -= Q;
	z += Q & -(z >> 31);
	return z;
}

/*
 * Compute NTT on a ring element.
 */
// __global__ void mq_NTT_gpu(uint16_t *a, unsigned logn)
// {
// 	size_t t, m;
// 	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	
// 	t = N;
// 	for (m = 1; m < N; m <<= 1) {
// 		uint32_t ht;

// 		ht = t >> 1;		
// 		uint32_t j;
// 		uint32_t s;

// 		s = GMb[m + tid/ht];
// 		j = bid*N + tid%ht + (tid/ht)*t;
// 		uint32_t u, v;

// 		u = a[j];
// 		v = mq_montymul(a[j + ht], s);

// 		a[j] = (uint16_t)mq_add(u, v);
// 		a[j + ht] = (uint16_t)mq_sub(u, v);
// 		t = ht;
// 		__syncthreads();
// 	}
// }

// wklee, shared memory version
__global__ void mq_NTT_gpu(uint16_t *a, unsigned logn)
{
	size_t t, m;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	__shared__ uint16_t s_a[N];

	s_a[tid] = a[bid*N + tid];
	s_a[tid+256] = a[bid*N + tid+256];
	__syncthreads();

	t = N;
	for (m = 1; m < N; m <<= 1) {
		uint32_t ht;

		ht = t >> 1;		
		uint32_t j;
		uint32_t s;

		s = GMb[m + tid/ht];
		j = tid%ht + (tid/ht)*t;
		uint32_t u, v;

		u = s_a[j];
		v = mq_montymul(s_a[j + ht], s);

		s_a[j] = (uint16_t)mq_add(u, v);
		s_a[j + ht] = (uint16_t)mq_sub(u, v);
		t = ht;
		__syncthreads();
	}

	a[bid*N + tid] = s_a[tid];
	a[bid*N + tid+256] = s_a[tid+256];
}

/*
 * Convert a polynomial (mod q) to Montgomery representation.
 */
__global__ void mq_poly_tomonty(uint16_t *f)
{
	// size_t u;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	f[bid*N + tid] = (uint16_t)mq_montymul(f[bid*N + tid], R2);
	f[bid*N + N/2 + tid] = (uint16_t)mq_montymul(f[bid*N + N/2 + tid], R2);	

}


/*
 * Multiply two polynomials together (NTT representation, and using
 * a Montgomery multiplication). Result f*g is written over f.
 */
__global__  void mq_poly_montymul_ntt_gpu(uint16_t *f, const uint16_t *g)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	f[bid*N + tid] = (uint16_t)mq_montymul(f[bid*N + tid], g[bid*N + tid]);
}

// __global__  void mq_iNTT_gpu(uint16_t *a)
// {
// 	uint32_t t, m;
// 	uint32_t ni = 128;	// pre-computed from falcon512fpu
// 	uint32_t u, v, w;
// 	uint32_t hm, dt;
// 	uint32_t tid = threadIdx.x, bid = blockIdx.x;
// 	t = 1;
// 	for (m = N; m > 1; m >>= 1) {
// 		hm = m >> 1;
// 		dt = t << 1;
// 		uint32_t j, s;
						
// 		s = iGMb[hm + tid/t];
// 		j = bid*N + tid%t + (tid/t)*dt;
// 		u = a[j];
// 		v = a[j + t];
// 		a[j] = (uint16_t)mq_add(u, v);
// 		w = mq_sub(u, v);
// 		a[j + t] = (uint16_t)mq_montymul(w, s);

// 		t = dt;	
// 		__syncthreads();
// 	}
	
// 	a[bid*N + tid] = (uint16_t)mq_montymul(a[bid*N + tid], ni);
// 	a[bid*N + N/2 + tid] = (uint16_t)mq_montymul(a[bid*N +  N/2 + tid], ni);	
// }
// wklee, shared memory version, slightly faster
__global__  void mq_iNTT_gpu(uint16_t *a)
{
	uint32_t t, m;
	uint32_t ni = 128;	// pre-computed from falcon512fpu
	uint32_t u, v, w;
	uint32_t hm, dt;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	t = 1;
	__shared__ uint16_t s_a[N];

	s_a[tid] = a[bid*N + tid];
	s_a[tid+256] = a[bid*N + tid+256];
	__syncthreads();

	for (m = N; m > 1; m >>= 1) {
		hm = m >> 1;
		dt = t << 1;
		uint32_t j, s;
						
		s = iGMb[hm + tid/t];
		j = tid%t + (tid/t)*dt;
		u = s_a[j];
		v = s_a[j + t];
		s_a[j] = (uint16_t)mq_add(u, v);
		w = mq_sub(u, v);
		s_a[j + t] = (uint16_t)mq_montymul(w, s);

		t = dt;	
		__syncthreads();
	}
	
	a[bid*N + tid] = (uint16_t)mq_montymul(s_a[tid], ni);
	a[bid*N + N/2 + tid] = (uint16_t)mq_montymul(s_a[N/2 + tid], ni);	
}

/*
 * Subtract polynomial g from polynomial f.
 */
__global__ void mq_poly_sub(uint16_t *f, const uint16_t *g)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	f[bid*N + tid] = (uint16_t)mq_sub(f[bid*N + tid], g[bid*N + tid]);
}

// __global__ void comb_all_kernels(uint16_t *a, int16_t *s2, uint16_t *g, uint16_t *h)
// {
//     uint32_t w;
//     uint32_t t, m;
//     uint32_t u, v;
//     uint32_t tid = threadIdx.x, bid = blockIdx.x;
// 	__shared__ uint16_t s_a[N];

// 	// reduce_s2
//     w = (uint32_t)s2[bid*N + tid];
//     w += Q & -(w >> 31);
//     a[bid*N + tid] = (uint16_t)w;
//     w = (uint32_t)s2[bid*N + tid + 256];
//     w += Q & -(w >> 31);
//     a[bid*N + tid + 256] = (uint16_t)w;
//     // __syncthreads();

//     // NTT
// 	s_a[tid] = a[bid*N + tid];
// 	s_a[tid+256] = a[bid*N + tid+256];
// 	__syncthreads();

// 	t = N;
// 	for (m = 1; m < N; m <<= 1) {
// 		uint32_t ht;

// 		ht = t >> 1;		
// 		uint32_t j;
// 		uint32_t s;

// 		s = GMb[m + tid/ht];
// 		j = tid%ht + (tid/ht)*t;
		

// 		u = s_a[j];
// 		v = mq_montymul(s_a[j + ht], s);

// 		s_a[j] = (uint16_t)mq_add(u, v);
// 		s_a[j + ht] = (uint16_t)mq_sub(u, v);
// 		t = ht;
// 		__syncthreads();
// 	}

// 	// mq_poly_montymul_ntt_gpu
// 	s_a[tid] = (uint16_t)mq_montymul(s_a[tid], g[bid*N + tid]);
// 	s_a[256 + tid] = (uint16_t)mq_montymul(s_a[256 + tid], g[bid*N + 256 + tid]);	
// 	__syncthreads();
// 	// iNTT
// 	uint32_t ni = 128;	// pre-computed from falcon512fpu	
// 	uint32_t hm, dt;
// 	t = 1;	

// 	for (m = N; m > 1; m >>= 1) {
// 		hm = m >> 1;
// 		dt = t << 1;
// 		uint32_t j, s;
						
// 		s = iGMb[hm + tid/t];
// 		j = tid%t + (tid/t)*dt;
// 		u = s_a[j];
// 		v = s_a[j + t];
// 		s_a[j] = (uint16_t)mq_add(u, v);
// 		w = mq_sub(u, v);
// 		s_a[j + t] = (uint16_t)mq_montymul(w, s);

// 		t = dt;	
// 		__syncthreads();
// 	}
	
// 	s_a[tid] = (uint16_t)mq_montymul(s_a[tid], ni);
// 	s_a[N/2 + tid] = (uint16_t)mq_montymul(s_a[N/2 + tid], ni);		
// 	__syncthreads();
// 	//mq_poly_sub
// 	s_a[tid] = (uint16_t)mq_sub(s_a[tid], h[bid*N + tid]);
// 	s_a[N/2 + tid] = (uint16_t)mq_sub(s_a[tid + N/2], h[bid*N + N/2 + tid]);
// 	__syncthreads();
// 	// norm_s2
//     w = (int32_t)s_a[tid];
//     w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
//     a[bid*N + tid] = (int16_t)w;
//     w = (int32_t)s_a[N/2 + tid];
//     w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
//     a[bid*N + N/2 + tid] = (int16_t)w;
// }


__global__ void comb_all_kernels(uint16_t *a, int16_t *s2, uint16_t *g, uint16_t *h)
{
    uint32_t w;
    uint32_t t, m;
    uint32_t u, v;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
	__shared__ uint16_t s_a[N];

	// reduce_s2
    w = (uint32_t)s2[bid*N + tid];
    w += Q & -(w >> 31);
    a[bid*N + tid] = (uint16_t)w;
    w = (uint32_t)s2[bid*N + tid + 256];
    w += Q & -(w >> 31);
    a[bid*N + tid + 256] = (uint16_t)w;
    // __syncthreads();

    // NTT
	s_a[tid] = a[bid*N + tid];
	s_a[tid+256] = a[bid*N + tid+256];
	__syncthreads();

	t = N;
	for (m = 1; m < N; m <<= 1) {
		uint32_t ht;

		ht = t >> 1;		
		uint32_t j;
		uint32_t s;

		s = GMb[m + tid/ht];
		j = tid%ht + (tid/ht)*t;
		

		u = s_a[j];
		v = mq_montymul(s_a[j + ht], s);

		s_a[j] = (uint16_t)mq_add(u, v);
		s_a[j + ht] = (uint16_t)mq_sub(u, v);
		t = ht;
		__syncthreads();
	}    
	// mq_poly_montymul_ntt_gpu
	s_a[tid] = (uint16_t)mq_montymul(s_a[tid], g[bid*N + tid]);
	s_a[256 + tid] = (uint16_t)mq_montymul(s_a[256 + tid], g[bid*N + 256 + tid]);	
	__syncthreads();

	// iNTT
	uint32_t ni = 128;	// pre-computed from falcon512fpu
	uint32_t hm, dt;
	t = 1;	

	for (m = N; m > 1; m >>= 1) {
		hm = m >> 1;
		dt = t << 1;
		uint32_t j, s;
						
		s = iGMb[hm + tid/t];
		j = tid%t + (tid/t)*dt;
		u = s_a[j];
		v = s_a[j + t];
		s_a[j] = (uint16_t)mq_add(u, v);
		w = mq_sub(u, v);
		s_a[j + t] = (uint16_t)mq_montymul(w, s);

		t = dt;	
		__syncthreads();
	}
	
	s_a[tid] = (uint16_t)mq_montymul(s_a[tid], ni);
	s_a[N/2 + tid] = (uint16_t)mq_montymul(s_a[N/2 + tid], ni);		

	//mq_poly_sub
	s_a[tid] = (uint16_t)mq_sub(s_a[tid], h[bid*N + tid]);
	s_a[N/2 + tid] = (uint16_t)mq_sub(s_a[tid + N/2], h[bid*N + N/2 + tid]);
	// norm_s2
    w = (int32_t)s_a[tid];
    w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    a[bid*N + tid] = (int16_t)w;
    w = (int32_t)s_a[N/2 + tid];
    w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    a[bid*N + N/2 + tid] = (int16_t)w;
}
