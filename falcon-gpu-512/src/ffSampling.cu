#include "../include/fft.cuh"
#include "../include/ffSampling.cuh"
#include "../include/consts.cuh"
#include "../include/rng.cuh"

__device__ void poly_merge_fft_s(fpr *f, const fpr *f0, const fpr *f1, unsigned logn)
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

__global__ void poly_merge_fft_p(fpr *f, const fpr *f0, const fpr *f1, unsigned logn)
{
	size_t n, hn, qn;
	uint32_t tid = threadIdx.x;
	n = (size_t)1 << logn;
	hn = n >> 1;
	qn = hn >> 1;

	/*
	 * An extra copy to handle the special case logn = 1.
	 */
	f[0] = f0[0];
	f[hn] = f1[0];
	fpr a_re, a_im, b_re, b_im;
	fpr t_re, t_im;

	a_re = f0[tid];
	a_im = f0[tid + qn];
	FPC_MUL(b_re, b_im, f1[tid], f1[tid + qn],
		fpr_gm_tab[((tid + hn) << 1) + 0],
		fpr_gm_tab[((tid + hn) << 1) + 1]);
	FPC_ADD(t_re, t_im, a_re, a_im, b_re, b_im);
	f[(tid << 1) + 0] = t_re;
	f[(tid << 1) + 0 + hn] = t_im;
	FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
	f[(tid << 1) + 1] = t_re;
	f[(tid << 1) + 1 + hn] = t_im;
}

__device__ void poly_LDL_fft_s(const fpr *g00, fpr *g01, fpr *g11, unsigned logn)
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

__global__ void poly_LDL_fft_p(const fpr *g00, fpr *g01, fpr *g11, unsigned logn)
{
	size_t n, hn;
	uint32_t tid = threadIdx.x;

	n = (size_t)1 << logn;
	hn = n >> 1;

	fpr g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
	fpr mu_re, mu_im;

	g00_re = g00[tid];
	g00_im = g00[tid + hn];
	g01_re = g01[tid];
	g01_im = g01[tid + hn];
	g11_re = g11[tid];
	g11_im = g11[tid + hn];
	FPC_DIV(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
	FPC_MUL(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg(g01_im));
	FPC_SUB(g11[tid], g11[tid + hn], g11_re, g11_im, g01_re, g01_im);
	g01[tid] = mu_re;
	g01[tid + hn] = fpr_neg(mu_im);
}


__device__ void poly_split_fft_s(fpr *f0, fpr *f1, const fpr *f, unsigned logn)
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

__device__  void poly_mul_fft_s(
	fpr *a, const fpr *b, unsigned logn)
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
		FPC_MUL(a[u], a[u + hn], a_re, a_im, b_re, b_im);
	}
}

__global__  void poly_mul_add_fft_p(
	fpr *a, const fpr *b, fpr *c, unsigned logn)
{
	size_t n, hn;
	uint32_t tid = threadIdx.x;

	n = (size_t)1 << logn;
	hn = n >> 1;

	fpr a_re, a_im, b_re, b_im;
	a_re = a[tid];
	a_im = a[tid + hn];
	b_re = b[tid];
	b_im = b[tid + hn];
	FPC_MUL(a[tid], a[tid + hn], a_re, a_im, b_re, b_im);
	// __syncthreads();
	c[tid] = fpr_add(a[tid], c[tid]);
	c[tid + hn] = fpr_add(a[tid + hn], c[tid + hn]);
	// __syncthreads();
}

/* see inner.h */
__device__  void poly_add_s(fpr *a, const fpr *b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_add(a[u], b[u]);
	}
}

__global__  void poly_add_p(fpr *a, const fpr *b, unsigned logn)
{
	uint32_t tid = threadIdx.x;	
	a[tid] = fpr_add(a[tid], b[tid]);
}

/* see inner.h */
__device__  void poly_sub_s(fpr *a, const fpr *b, unsigned logn)
{
	size_t n, u;

	n = (size_t)1 << logn;
	for (u = 0; u < n; u ++) {
		a[u] = fpr_sub(a[u], b[u]);
	}
}

/*
 * Get a 64-bit random value from a PRNG.
 */
__device__ uint64_t prng_get_u64(prng_s *p)
{
	size_t u;

	/*
	 * If there are less than 9 bytes in the buffer, we refill it.
	 * This means that we may drop the last few bytes, but this allows
	 * for faster extraction code. Also, it means that we never leave
	 * an empty buffer.
	 */
	u = p->ptr;
	if (u >= (sizeof p->buf.d) - 9) {
		prng_refill_s(p);
		u = 0;
	}
	p->ptr = u + 8;

	/*
	 * On systems that use little-endian encoding and allow
	 * unaligned accesses, we can simply read the data where it is.
	 */
	return (uint64_t)p->buf.d[u + 0]
		| ((uint64_t)p->buf.d[u + 1] << 8)
		| ((uint64_t)p->buf.d[u + 2] << 16)
		| ((uint64_t)p->buf.d[u + 3] << 24)
		| ((uint64_t)p->buf.d[u + 4] << 32)
		| ((uint64_t)p->buf.d[u + 5] << 40)
		| ((uint64_t)p->buf.d[u + 6] << 48)
		| ((uint64_t)p->buf.d[u + 7] << 56);
}

/*
 * Get an 8-bit random value from a PRNG.
 */
__device__ unsigned prng_get_u8(prng_s *p)
{
	unsigned v;

	v = p->buf.d[p->ptr ++];
	if (p->ptr == sizeof p->buf.d) {
		prng_refill_s(p);
	}
	return v;
}

/*
 * Sample an integer value along a half-gaussian distribution centered
 * on zero and standard deviation 1.8205, with a precision of 72 bits.
 */
__device__ int prng_sgaussian0_sampler(prng_s *p)
{

	static const uint32_t dist[] = {
		10745844u,  3068844u,  3741698u,
		 5559083u,  1580863u,  8248194u,
		 2260429u, 13669192u,  2736639u,
		  708981u,  4421575u, 10046180u,
		  169348u,  7122675u,  4136815u,
		   30538u, 13063405u,  7650655u,
		    4132u, 14505003u,  7826148u,
		     417u, 16768101u, 11363290u,
		      31u,  8444042u,  8086568u,
		       1u, 12844466u,   265321u,
		       0u,  1232676u, 13644283u,
		       0u,    38047u,  9111839u,
		       0u,      870u,  6138264u,
		       0u,       14u, 12545723u,
		       0u,        0u,  3104126u,
		       0u,        0u,    28824u,
		       0u,        0u,      198u,
		       0u,        0u,        1u
	};

	uint32_t v0, v1, v2, hi;
	uint64_t lo;
	size_t u;
	int z;

	/*
	 * Get a random 72-bit value, into three 24-bit limbs v0..v2.
	 */
	lo = prng_get_u64(p);
	hi = prng_get_u8(p);
	v0 = (uint32_t)lo & 0xFFFFFF;
	v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
	v2 = (uint32_t)(lo >> 48) | (hi << 16);

	/*
	 * Sampled value is z, such that v0..v2 is lower than the first
	 * z elements of the table.
	 */
	z = 0;
	for (u = 0; u < (sizeof dist) / sizeof(dist[0]); u += 3) {
		uint32_t w0, w1, w2, cc;

		w0 = dist[u + 2];
		w1 = dist[u + 1];
		w2 = dist[u + 0];
		cc = (v0 - w0) >> 31;
		cc = (v1 - w1 - cc) >> 31;
		cc = (v2 - w2 - cc) >> 31;
		z += (int)cc;
	}
	return z;

}

/*
 * Sample a bit with probability exp(-x) for some x >= 0.
 */
__device__  int BerExp(prng_s *p, fpr x, fpr ccs)
{
	int s, i;
	fpr r;
	uint32_t sw, w;
	uint64_t z;

	/*
	 * Reduce x modulo log(2): x = s*log(2) + r, with s an integer,
	 * and 0 <= r < log(2). Since x >= 0, we can use fpr_trunc().
	 */
	s = (int)fpr_trunc(fpr_mul(x, fpr_inv_log2));
	r = fpr_sub(x, fpr_mul(fpr_of(s), fpr_log2));

	/*
	 * It may happen (quite rarely) that s >= 64; if sigma = 1.2
	 * (the minimum value for sigma), r = 0 and b = 1, then we get
	 * s >= 64 if the half-Gaussian produced a z >= 13, which happens
	 * with probability about 0.000000000230383991, which is
	 * approximatively equal to 2^(-32). In any case, if s >= 64,
	 * then BerExp will be non-zero with probability less than
	 * 2^(-64), so we can simply saturate s at 63.
	 */
	sw = (uint32_t)s;
	sw ^= (sw ^ 63) & -((63 - sw) >> 31);
	s = (int)sw;

	/*
	 * Compute exp(-r); we know that 0 <= r < log(2) at this point, so
	 * we can use fpr_expm_p63(), which yields a result scaled to 2^63.
	 * We scale it up to 2^64, then right-shift it by s bits because
	 * we really want exp(-x) = 2^(-s)*exp(-r).
	 *
	 * The "-1" operation makes sure that the value fits on 64 bits
	 * (i.e. if r = 0, we may get 2^64, and we prefer 2^64-1 in that
	 * case). The bias is negligible since fpr_expm_p63() only computes
	 * with 51 bits of precision or so.
	 */
	z = ((fpr_expm_p63(r, ccs) << 1) - 1) >> s;

	/*
	 * Sample a bit with probability exp(-x). Since x = s*log(2) + r,
	 * exp(-x) = 2^-s * exp(-r), we compare lazily exp(-x) with the
	 * PRNG output to limit its consumption, the sign of the difference
	 * yields the expected result.
	 */
	i = 64;
	do {
		i -= 8;
		w = prng_get_u8(p) - ((uint32_t)(z >> i) & 0xFF);
	} while (!w && i > 0);
	return (int)(w >> 31);
}

/*
 * The sampler produces a random integer that follows a discrete Gaussian
 * distribution, centered on mu, and with standard deviation sigma. The
 * provided parameter isigma is equal to 1/sigma.
 *
 * The value of sigma MUST lie between 1 and 2 (i.e. isigma lies between
 * 0.5 and 1); in Falcon, sigma should always be between 1.2 and 1.9.
 */

__device__ int sampler(sampler_context_s *spc, fpr mu, fpr isigma)
{
	// sampler_context_s *spc;
	int s;
	fpr r, dss, ccs;
	// spc = ctx;	
	/*
	 * Center is mu. We compute mu = s + r where s is an integer
	 * and 0 <= r < 1.
	 */
	s = (int)fpr_floor(mu);
	r = fpr_sub(mu, fpr_of(s));

	/*
	 * dss = 1/(2*sigma^2) = 0.5*(isigma^2).
	 */
	dss = fpr_half(fpr_sqr(isigma));

	/*
	 * ccs = sigma_min / sigma = sigma_min * isigma.
	 */
	// ccs = fpr_mul(isigma, spc->sigma_min);
	 ccs = fpr_mul(isigma, fpr_sigma_min[9]);

	/*
	 * We now need to sample on center r.
	 */
	for (;;) {
		int z0, z, b;
		fpr x;

		/*
		 * Sample z for a Gaussian distribution. Then get a
		 * random bit b to turn the sampling into a bimodal
		 * distribution: if b = 1, we use z+1, otherwise we
		 * use -z. We thus have two situations:
		 *
		 *  - b = 1: z >= 1 and sampled against a Gaussian
		 *    centered on 1.
		 *  - b = 0: z <= 0 and sampled against a Gaussian
		 *    centered on 0.
		 */
		z0 = prng_sgaussian0_sampler(&spc->p);
		b = (int)prng_get_u8(&spc->p) & 1;
		z = b + ((b << 1) - 1) * z0;

		/*
		 * Rejection sampling. We want a Gaussian centered on r;
		 * but we sampled against a Gaussian centered on b (0 or
		 * 1). But we know that z is always in the range where
		 * our sampling distribution is greater than the Gaussian
		 * distribution, so rejection works.
		 *
		 * We got z with distribution:
		 *    G(z) = exp(-((z-b)^2)/(2*sigma0^2))
		 * We target distribution:
		 *    S(z) = exp(-((z-r)^2)/(2*sigma^2))
		 * Rejection sampling works by keeping the value z with
		 * probability S(z)/G(z), and starting again otherwise.
		 * This requires S(z) <= G(z), which is the case here.
		 * Thus, we simply need to keep our z with probability:
		 *    P = exp(-x)
		 * where:
		 *    x = ((z-r)^2)/(2*sigma^2) - ((z-b)^2)/(2*sigma0^2)
		 *
		 * Here, we scale up the Bernouilli distribution, which
		 * makes rejection more probable, but makes rejection
		 * rate sufficiently decorrelated from the Gaussian
		 * center and standard deviation that the whole sampler
		 * can be said to be constant-time.
		 */
		x = fpr_mul(fpr_sqr(fpr_sub(fpr_of(z), r)), dss);
		x = fpr_sub(x, fpr_mul(fpr_of(z0 * z0), fpr_inv_2sqrsigma0));
		if (BerExp(&spc->p, x, ccs)) {
			/*
			 * Rejection sampling was centered on r, but the
			 * actual center is mu = s + r.
			 */
			return s + z;
		}
	}
}

// __global__ void ffSampling_fft_dyntree(fpr *t0, fpr *t1, fpr *g00, fpr *g01, fpr *g11,	unsigned orig_logn, unsigned logn, fpr *tmp, uint64_t *scA, uint64_t *scdptr)
// {
// 	size_t n, hn, i;
// 	STACK stack[LOGN + 1];	//orig_logn + 1
// 	unsigned stack_top = 0;
// 	uint32_t bid = blockIdx.x;
// 	stack[0].t0 = t0 + bid*10*N;
// 	stack[0].t1 = t1 + bid*10*N;
// 	stack[0].g00 = g00 + bid*10*N;
// 	stack[0].g01 = g01 + bid*10*N;
// 	stack[0].g11 = g11 + bid*10*N;
// 	stack[0].logn = logn;
// 	stack[0].tmp = tmp + bid*10*N;
// 	stack[0].z0 = NULL;
// 	stack[0].z1 = NULL;

// 	__shared__ inner_shake256_context_s rng;	
// 	__shared__ sampler_context_s samp_ctx;
// 	samp_ctx.sigma_min = fpr_sigma_min[logn];	
// 	samp_ctx.p.ptr = 0;
// 	samp_ctx.p.type = 0;
// 	rng.dptr = scdptr[bid];
//     for(i=0; i<25; i++) rng.st.A[i] = scA[i];

// 	prng_init_s(&samp_ctx.p, &rng);

// 	while (1)
// 	{
// 		/*
// 		 * Deepest level: the LDL tree leaf value is just g00 (the
// 		 * array has length only 1 at this point); we normalize it
// 		 * with regards to sigma, then use it for sampling.
// 		 */
// 		if (stack[stack_top].logn == 0) {
// 			fpr leaf;
// 			// printf(".");
// 			leaf = stack[stack_top].g00[0];
// 			leaf = fpr_mul(fpr_sqrt(leaf), fpr_inv_sigma[orig_logn]);
// 			stack[stack_top].t0[0] = fpr_of(sampler(&samp_ctx, stack[stack_top].t0[0], leaf));
// 			stack[stack_top].t1[0] = fpr_of(sampler(&samp_ctx, stack[stack_top].t1[0], leaf));
			
// 			if (stack[--stack_top].z0 == NULL)
// 			{
// 				poly_merge_fft_s(stack[stack_top].tmp + 4, stack[stack_top].z1, stack[stack_top].z1 + 1, 1);				
// 			}
// 			else
// 			{
// 				poly_merge_fft_s(stack[stack_top].t0, stack[stack_top].z0, stack[stack_top].z0 + 1, 1);
// 			}
// 		}
// 		else
// 		{
// 			n = (size_t)1 << stack[stack_top].logn;
// 			hn = n >> 1;

// 			if (stack[stack_top].z1 == NULL)
// 			{
// 				/*
// 				 * Decompose G into LDL. We only need d00 (identical to g00),
// 				 * d11, and l10; we do that in place.
// 				 */
// #ifdef DYN_PAR				
// 				if(stack[stack_top].logn>=8)
// 				{	
// 					poly_LDL_fft_p<<<1, 1 << (stack[stack_top].logn-1)>>>(stack[stack_top].g00, stack[stack_top].g01, stack[stack_top].g11, stack[stack_top].logn);
// 					cudaDeviceSynchronize() ;
// 				}
// 				else{
// 					poly_LDL_fft_s(stack[stack_top].g00, stack[stack_top].g01, stack[stack_top].g11, stack[stack_top].logn);
// 				}
// #else
// 				poly_LDL_fft_s(stack[stack_top].g00, stack[stack_top].g01, stack[stack_top].g11, stack[stack_top].logn);
// #endif						
// 				/*
// 				 * Split d00 and d11 and expand them into half-size quasi-cyclic
// 				 * Gram matrices. We also save l10 in tmp[].
// 				 */
// 				poly_split_fft_s(stack[stack_top].tmp, stack[stack_top].tmp + hn, stack[stack_top].g00, stack[stack_top].logn);
// 				for (i = 0; i < n; ++i)
// 				{
// 					stack[stack_top].g00[i].v = stack[stack_top].tmp[i].v;
// 				}
// 				poly_split_fft_s(stack[stack_top].tmp, stack[stack_top].tmp + hn, stack[stack_top].g11, stack[stack_top].logn);
// 				for (i = 0; i < n; ++i)
// 				{
// 					stack[stack_top].g11[i].v = stack[stack_top].tmp[i].v;
// 					stack[stack_top].tmp[i].v = stack[stack_top].g01[i].v;
// 				}				
// 				for (i = 0; i < hn; ++i)
// 				{
// 					stack[stack_top].g01[i].v = stack[stack_top].g00[i].v;
// 					stack[stack_top].g01[i+hn].v = stack[stack_top].g11[i].v;
// 				}				

// 				/*
// 				 * The half-size Gram matrices for the recursive LDL tree
// 				 * building are now:
// 				 *   - left sub-tree: g00, g00+hn, g01
// 				 *   - right sub-tree: g11, g11+hn, g01+hn
// 				 * l10 is in tmp[].
// 				 */
				 
// 				/*
// 				 * We split t1 and use the first recursive call on the two
// 				 * halves, using the right sub-tree. The result is merged
// 				 * back into tmp + 2*n.
// 				 */
// 				stack[stack_top].z1 = stack[stack_top].tmp + n;
// 				poly_split_fft_s(stack[stack_top].z1, stack[stack_top].z1 + hn, stack[stack_top].t1, stack[stack_top].logn);

// 				stack[stack_top + 1].t0 = stack[stack_top].z1;
// 				stack[stack_top + 1].t1 = stack[stack_top].z1 + hn;
// 				stack[stack_top + 1].g00 = stack[stack_top].g11;
// 				stack[stack_top + 1].g01 = stack[stack_top].g11 + hn;
// 				stack[stack_top + 1].g11 = stack[stack_top].g01 + hn;
// 				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
// 				stack[stack_top + 1].tmp = stack[stack_top].z1 + n;
// 				stack[stack_top + 1].z0 = NULL;
// 				stack[++stack_top].z1 = NULL;
// 			}
// 			else if (stack[stack_top].z0 == NULL)
// 			{
// 				/*
// 				 * Compute tb0 = t0 + (t1 - z1) * l10.
// 				 * At that point, l10 is in tmp, t1 is unmodified, and z1 is
// 				 * in tmp + (n << 1). The buffer in z1 is free.
// 				 *
// 				 * In the end, z1 is written over t1, and tb0 is in t0.
// 				 */
// 				for (i = 0; i < n; ++i)
// 				{
// 					stack[stack_top].z1[i].v = stack[stack_top].t1[i].v;
// 				}				 
// 				poly_sub_s(stack[stack_top].z1, stack[stack_top].tmp + (n << 1), stack[stack_top].logn);
// 				for (i = 0; i < n; ++i)
// 				{
// 					stack[stack_top].t1[i].v = stack[stack_top].tmp[i + (n << 1)].v;
// 				}				
// #ifdef DYN_PAR				
// 				if(stack[stack_top].logn>=8)
// 				{					
// 					poly_mul_add_fft_p<<<1,(size_t)1 << (stack[stack_top].logn-1)>>>(stack[stack_top].tmp, stack[stack_top].z1, stack[stack_top].t0, stack[stack_top].logn);			
// 					cudaDeviceSynchronize() ;
// 				}
// 				else{
// 					poly_mul_fft_s(stack[stack_top].tmp, stack[stack_top].z1, stack[stack_top].logn);
// 					poly_add_s(stack[stack_top].t0, stack[stack_top].tmp, stack[stack_top].logn);
// 				}
// #else					
// 				poly_mul_fft_s(stack[stack_top].tmp, stack[stack_top].z1, stack[stack_top].logn);
// 				poly_add_s(stack[stack_top].t0, stack[stack_top].tmp, stack[stack_top].logn);
// #endif		
// 				/*
// 				 * Second recursive invocation, on the split tb0 (currently in t0)
// 				 * and the left sub-tree.
// 				 */
// 				stack[stack_top].z0 = stack[stack_top].tmp;
// 				poly_split_fft_s(stack[stack_top].z0, stack[stack_top].z0 + hn, stack[stack_top].t0, stack[stack_top].logn);

// 				stack[stack_top + 1].t0 = stack[stack_top].z0;
// 				stack[stack_top + 1].t1 = stack[stack_top].z0 + hn;
// 				stack[stack_top + 1].g00 = stack[stack_top].g00;
// 				stack[stack_top + 1].g01 = stack[stack_top].g00 + hn;
// 				stack[stack_top + 1].g11 = stack[stack_top].g01;
// 				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
// 				stack[stack_top + 1].tmp = stack[stack_top].z0 + n;
// 				stack[stack_top + 1].z0 = NULL;
// 				stack[++stack_top].z1 = NULL;
// 			}
// 			else
// 			{
// 				if (stack[stack_top].logn == orig_logn)
// 				{
// 					return;
// 				}
// 				else
// 				{
// 					if (stack[--stack_top].z0 == NULL)
// 					{
// // #ifdef DYN_PAR							
// // 						if(stack[stack_top].logn>=7)
// // 						{
// // 							poly_merge_fft_p<<<1,1 << (stack[stack_top].logn-2)>>>(stack[stack_top].tmp + (n << 2), stack[stack_top].z1, stack[stack_top].z1 + n, stack[stack_top].logn);
// // 							cudaDeviceSynchronize() ;
// // 						}
// // 						else{

// // 						// printf("%u \n", stack[stack_top].logn);
// // 							poly_merge_fft_s(stack[stack_top].tmp + (n << 2), stack[stack_top].z1, stack[stack_top].z1 + n, stack[stack_top].logn);
// // 						}
// // #else				
// 						poly_merge_fft_s(stack[stack_top].tmp + (n << 2), stack[stack_top].z1, stack[stack_top].z1 + n, stack[stack_top].logn);

// // #endif							

// 					}
// 					else
// 					{
// // #ifdef DYN_PAR							
// // 						if(stack[stack_top].logn>=10)
// // 						{
// // 							poly_merge_fft_p<<<1,1 << (stack[stack_top].logn-2)>>>(stack[stack_top].t0, stack[stack_top].z0, stack[stack_top].z0 + n, stack[stack_top].logn);
// // 							cudaDeviceSynchronize() ;
// // 						}
// // 						else{
// 						// 	poly_merge_fft_s(stack[stack_top].t0, stack[stack_top].z0, stack[stack_top].z0 + n, stack[stack_top].logn);
// 						// }
// // #else							
// 							poly_merge_fft_s(stack[stack_top].t0, stack[stack_top].z0, stack[stack_top].z0 + n, stack[stack_top].logn);
// 					}
// 				}
// 			}
// 		}
// 	}
// }


 
__global__ void ffSampling_fft_dyntree(fpr *t0, fpr *t1, fpr *g00, fpr *g01, fpr *g11,	unsigned orig_logn, unsigned logn, fpr *tmp, uint64_t *scA, uint64_t *scdptr)
{
	size_t n, hn, i;
	STACK stack[LOGN + 1];	//orig_logn + 1
	unsigned stack_top = 0;
	uint32_t bid = blockIdx.x;

	stack[0].t0 = t0 + bid*10*N;
	stack[0].g00 = g00 + bid*10*N;
	stack[0].g11 = g11 + bid*10*N;
	stack[0].logn = logn;
	stack[0].is_z0 = 0;
	stack[0].is_z1 = 0;

	__shared__ inner_shake256_context_s rng;	
	__shared__ sampler_context_s samp_ctx;
	samp_ctx.sigma_min = fpr_sigma_min[logn];	
	samp_ctx.p.ptr = 0;
	samp_ctx.p.type = 0;
	rng.dptr = scdptr[bid];
    for(i=0; i<25; i++) rng.st.A[i] = scA[i];

	prng_init_s(&samp_ctx.p, &rng);

	while (1)
	{
		/*
		 * Deepest level: the LDL tree leaf value is just g00 (the
		 * array has length only 1 at this point); we normalize it
		 * with regards to sigma, then use it for sampling.
		 */
		if (stack[stack_top].logn == 0) {
			fpr leaf;

			leaf = stack[stack_top].g00[0];
			leaf = fpr_mul(fpr_sqrt(leaf), fpr_inv_sigma[orig_logn]);
			// stack[stack_top].t0[0] = fpr_of(samp(samp_ctx, stack[stack_top].t0[0], leaf));
			// stack[stack_top].t0[1] = fpr_of(samp(samp_ctx, stack[stack_top].t0[1], leaf));

			stack[stack_top].t0[0] = fpr_of(sampler(&samp_ctx, stack[stack_top].t0[0], leaf));
			stack[stack_top].t0[1] = fpr_of(sampler(&samp_ctx, stack[stack_top].t0[1], leaf));			
			
			if (!stack[--stack_top].is_z0)
			{
				poly_merge_fft_s(stack[stack_top].t0 + 8, stack[stack_top].t0 + 6, stack[stack_top].t0 + 7, 1);
			}
			else
			{
				poly_merge_fft_s(stack[stack_top].t0, stack[stack_top].t0 + 4, stack[stack_top].t0 + 5, 1);
			}
		}
		else
		{
			n = (size_t)1 << stack[stack_top].logn;
			hn = n >> 1;

			if (!stack[stack_top].is_z1)
			{
				/*
				 * Decompose G into LDL. We only need d00 (identical to g00),
				 * d11, and l10; we do that in place.
				 */
				poly_LDL_fft_s(stack[stack_top].g00, stack[stack_top].g00 + n, stack[stack_top].g11, stack[stack_top].logn);

				/*
				 * Split d00 and d11 and expand them into half-size quasi-cyclic
				 * Gram matrices. We also save l10 in tmp[].
				 */
				poly_split_fft_s(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].g00, stack[stack_top].logn);
				memcpy(stack[stack_top].g00, stack[stack_top].t0 + (n << 1), n * sizeof *(t0 + (n << 1)));
				poly_split_fft_s(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].g11, stack[stack_top].logn);
				memcpy(stack[stack_top].g11, stack[stack_top].t0 + (n << 1), n * sizeof *(t0 + (n << 1)));
				memcpy(stack[stack_top].t0 + (n << 1), stack[stack_top].g00 + n, n * sizeof *(g00 + n));
				memcpy(stack[stack_top].g00 + n, stack[stack_top].g00, hn * sizeof *g00);
				memcpy(stack[stack_top].g00 + n + hn, stack[stack_top].g11, hn * sizeof *g00);

				/*
				 * The half-size Gram matrices for the recursive LDL tree
				 * building are now:
				 *   - left sub-tree: g00, g00+hn, g01
				 *   - right sub-tree: g11, g11+hn, g01+hn
				 * l10 is in tmp[].
				 */
				 
				/*
				 * We split t1 and use the first recursive call on the two
				 * halves, using the right sub-tree. The result is merged
				 * back into tmp + 2*n.
				 */
				stack[stack_top].is_z1 = 1;
				poly_split_fft_s(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + 3 * n + hn, stack[stack_top].t0 + n, stack[stack_top].logn);

				stack[stack_top + 1].t0 = stack[stack_top].t0 + 3 * n;
				stack[stack_top + 1].g00 = stack[stack_top].g11;
				stack[stack_top + 1].g11 = stack[stack_top].g00 + n + hn;
				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
				stack[stack_top + 1].is_z0 = 0;
				stack[++stack_top].is_z1 = 0;
			}
			else if (!stack[stack_top].is_z0)
			{
				/*
				 * Compute tb0 = t0 + (t1 - z1) * l10.
				 * At that point, l10 is in tmp, t1 is unmodified, and z1 is
				 * in tmp + (n << 1). The buffer in z1 is free.
				 *
				 * In the end, z1 is written over t1, and tb0 is in t0.
				 */
				memcpy(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + n, n * sizeof *(t0 + n));
				poly_sub_s(stack[stack_top].t0 + 3 * n, stack[stack_top].t0 + (n << 2), stack[stack_top].logn);
				memcpy(stack[stack_top].t0 + n, stack[stack_top].t0 + (n << 2), n * sizeof *(t0 + (n << 1)));
				poly_mul_fft_s(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + 3 * n, stack[stack_top].logn);
				poly_add_s(stack[stack_top].t0, stack[stack_top].t0 + (n << 1), stack[stack_top].logn);

				/*
				 * Second recursive invocation, on the split tb0 (currently in t0)
				 * and the left sub-tree.
				 */
				stack[stack_top].is_z0 = 1;
				poly_split_fft_s(stack[stack_top].t0 + (n << 1), stack[stack_top].t0 + (n << 1) + hn, stack[stack_top].t0, stack[stack_top].logn);

				stack[stack_top + 1].t0 = stack[stack_top].t0 + (n << 1);
				stack[stack_top + 1].g00 = stack[stack_top].g00;
				stack[stack_top + 1].g11 = stack[stack_top].g00 + n;
				stack[stack_top + 1].logn = stack[stack_top].logn - 1;
				stack[stack_top + 1].is_z0 = 0;
				stack[++stack_top].is_z1 = 0;
			}
			else
			{
				if (stack[stack_top].logn == orig_logn)
				{
					return;
				}
				else
				{
					if (!stack[--stack_top].is_z0)
					{
						poly_merge_fft_s(stack[stack_top].t0 + (n << 3), stack[stack_top].t0 + 6 * n, stack[stack_top].t0 + 7 * n, stack[stack_top].logn);
					}
					else
					{
						poly_merge_fft_s(stack[stack_top].t0, stack[stack_top].t0 + (n << 2), stack[stack_top].t0 + 5 * n, stack[stack_top].logn);
					}
				}
			}
		}
	}
}