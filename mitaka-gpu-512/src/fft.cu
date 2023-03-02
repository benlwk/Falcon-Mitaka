#include "../include/fft.cuh"
#include "../include/consts.cuh"

// extern __device__ fpr fpr_mul(fpr x, fpr y);

// wklee, this is CPU version
void poly_add_s(fpr *a, const fpr *b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_add2(a[u], b[u]);
	}
}

void poly_sub_s(fpr *a, const fpr *b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_sub2(a[u], b[u]);
	}
}

void poly_LDL_fft_s(const fpr *g00, fpr *g01, fpr *g11, unsigned logn)
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
		FPC_DIV2(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
		FPC_MUL2(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg2(g01_im));
		FPC_SUB2(g11[u], g11[u + hn], g11_re, g11_im, g01_re, g01_im);
		g01[u] = mu_re;
		g01[u + hn] = fpr_neg2(mu_im);
	}
}

void poly_split_fft_s(fpr *f0, fpr *f1, const fpr *f, unsigned logn)
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

		FPC_ADD2(t_re, t_im, a_re, a_im, b_re, b_im);
		f0[u] = fpr_half2(t_re);
		f0[u + qn] = fpr_half2(t_im);

		FPC_SUB2(t_re, t_im, a_re, a_im, b_re, b_im);
		FPC_MUL2(t_re, t_im, t_re, t_im,
			fpr_gm_tab_s[((u + hn) << 1) + 0],
			fpr_neg2(fpr_gm_tab_s[((u + hn) << 1) + 1]));
		f1[u] = fpr_half2(t_re);
		f1[u + qn] = fpr_half2(t_im);
	}
}

void poly_merge_fft_s(fpr *f, const fpr *f0, const fpr *f1, unsigned logn)
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
		FPC_MUL2(b_re, b_im, f1[u], f1[u + qn],
			fpr_gm_tab_s[((u + hn) << 1) + 0],
			fpr_gm_tab_s[((u + hn) << 1) + 1]);
		FPC_ADD2(t_re, t_im, a_re, a_im, b_re, b_im);
		f[(u << 1) + 0] = t_re;
		f[(u << 1) + 0 + hn] = t_im;
		FPC_SUB2(t_re, t_im, a_re, a_im, b_re, b_im);
		f[(u << 1) + 1] = t_re;
		f[(u << 1) + 1 + hn] = t_im;
	}
}

// wklee, below are the parallel versions on GPU
__global__ void poly_set_g(fpr *out, const int16_t *in)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    out[bid*N + tid] = fpr_of(in[bid*N + tid]);   
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
    a_re = a[bid*N + tid];
    a_im = a[bid*N + tid + hn];
    a[bid*N + tid] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
    a[bid*N + tid + hn] = fpr_zero;
    
}

__global__ void poly_muladj_fft_g(fpr *a, const fpr *b)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    hn = N >> 1;

    fpr a_re, a_im, b_re, b_im;

    a_re = a[bid*N + tid];
    a_im = a[bid*N + tid + hn];
    b_re = b[bid*N + tid];
    b_im = fpr_neg(b[bid*N + tid + hn]);
    FPC_MUL(a[bid*N + tid], a[bid*N + tid + hn], a_re, a_im, b_re, b_im);
}

__global__ void poly_recenter(fpr* p) {
  // for(int i=0; i < MITAKA_D; ++i) {
  	uint32_t tid = threadIdx.x, bid = blockIdx.x;

    p[bid*N + tid].v = fmod(p[bid*N + tid].v, (double) MITAKA_Q);
    if(p[bid*N + tid].v > MITAKA_Q/2)
		p[bid*N + tid].v -= (double) MITAKA_Q;
    else if(p[bid*N + tid].v < -MITAKA_Q/2)
		p[bid*N + tid].v += (double) MITAKA_Q;
  // }
}

__global__ void poly_add_g(fpr *a, const fpr *b)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    a[bid*N + tid] = fpr_add(a[bid*N + tid], b[bid*N + tid]);    
}

__global__ void poly_sub_g(fpr *a, const fpr *b)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    a[bid*N + tid] = fpr_sub(a[bid*N + tid], b[bid*N + tid]);    
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

    hn = N >> 1;
    t = hn;
    m = 2;
    for (u = 0;  u < N/blockDim.x; u ++)    
        s_f[u*blockDim.x + tid].v = f[bid*in_s + u*blockDim.x + tid].v;
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
        f[bid*in_s + u*blockDim.x + tid].v = s_f[u*blockDim.x + tid].v;
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
	uint32_t n, hn, t, m, j;
	uint32_t u, tid = threadIdx.x, bid = blockIdx.x;	
	t = 1;
	m = N;
	hn = N >> 1;
	for (u = 9; u > 1; u --) {
		uint32_t hm, dt, i1 = 0, j1;
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

__global__ void poly_mul_fft(fpr *a, fpr *b)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    
    hn = N >> 1;
    fpr a_re, a_im, b_re, b_im;

    a_re = a[bid*N + tid];
    a_im = a[bid*N + tid + hn];
    b_re = b[bid*N + tid];
    b_im = b[bid*N + tid + hn];    
    FPC_MUL(a[bid*N + tid], a[bid*N + tid + hn], a_re, a_im, b_re, b_im);    
}

__global__ void poly_mulconst(fpr *a, fpr x)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    a[bid*N + tid] = fpr_mul(a[bid*N + tid], x);
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
    out[bid*N + tid] = in[bid*N + tid];
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

void poly_mul_fft_s(fpr *a, const fpr *b, unsigned logn)
{
	size_t n, hn, u;

	n = (size_t)1 << logn;
	hn = n >> 1;
	for (u = 0; u < hn; u ++) {
		fpr a_re, a_im, b_re, b_im;

		a_re = a[u];
		a_im = a[u + hn];
		b_re = b[u];
		b_im = b[u + hn];
		FPC_MUL2(a[u], a[u + hn], a_re, a_im, b_re, b_im);
	}
}
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

__global__ void check_norm(fpr* p1, fpr* p2, uint8_t *reject){
  	double s=0.0;
  	uint32_t bid = blockIdx.x;
  	for(int i=0; i < MITAKA_D; ++i)
    	s += p1[i].v*p1[i].v + p2[i].v*p2[i].v;  
    // printf("%u\t", s);
    if(s <= GAMMA_SQUARE) {
    	printf("reject signature %u\n", bid);
    	reject[bid] = 1;
    }
  	// return s <= GAMMA_SQUARE;
}

__global__ void poly_mul_fftx2(fpr *a, fpr *b, fpr *c, fpr *d)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    hn = N >> 1;
    fpr a_re, a_im, b_re, b_im;

    a_re = a[bid*N + tid];
    a_im = a[bid*N + tid + hn];
    b_re = b[bid*N + tid];
    b_im = b[bid*N + tid + hn];    
    FPC_MUL(a[bid*N + tid], a[bid*N + tid + hn], a_re, a_im, b_re, b_im);   

    a_re = c[bid*N + tid];
    a_im = c[bid*N + tid + hn];
    b_re = d[bid*N + tid];
    b_im = d[bid*N + tid + hn];    
    FPC_MUL(c[bid*N + tid], c[bid*N + tid + hn], a_re, a_im, b_re, b_im);    
}

__global__ void poly_mul_fft_add(fpr *a, fpr *b, fpr *c)
{
    uint32_t hn;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    
    hn = N >> 1;
    fpr a_re, a_im, b_re, b_im, res_a1, res_a2;

    a_re = a[bid*N + tid];
    a_im = a[bid*N + tid + hn];
    b_re = b[bid*N + tid];
    b_im = b[bid*N + tid + hn];    
    FPC_MUL(res_a1, res_a2, a_re, a_im, b_re, b_im);    

    c[bid*N + tid] = fpr_add(c[bid*N + tid], res_a1);    
    c[bid*N + N/2 + tid] = fpr_add(c[bid*N + N/2 +tid], res_a2);    
}