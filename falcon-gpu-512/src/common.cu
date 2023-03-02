#include "../include/common.cuh"
#include "../include/fft.cuh"

__global__ void check2(int16_t* s2tmp, fpr *t1)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    // s2tmp[bid*N + tid] = (int16_t)-fpr_rint(t1[bid*10*N + tid]);
    s2tmp[bid*N + tid] = (int16_t) - fpr_rint(t1[bid*10*N + tid]);
}

__global__ void check1(int16_t* s1tmp, fpr *t0, uint16_t *hm, uint32_t *sqn)
{
    uint32_t bid = blockIdx.x;
    uint32_t ng, u;
    sqn[bid] = 0;
    ng = 0;
    for (u = 0; u < N; u ++) {
        int32_t z;

        z = (int32_t)hm[bid*N + u] - (int32_t)fpr_rint(t0[bid*10*N + u]);
        sqn[bid] += (uint32_t)(z * z);
        ng |=sqn[bid];
        s1tmp[bid*10*N + u] = (int16_t)z;
    }
    sqn[bid] |= -(ng >> 31);
    // if(bid==0) printf("%u\n", sqn[bid]);
}

     /*  
     * Reduce s2 elements modulo q ([0..q-1] range).
     */
__global__ void reduce_s2(int16_t *s2, uint16_t *tt)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    uint32_t w;

    w = (uint32_t)s2[bid*N + tid];
    w += Q & -(w >> 31);
    tt[bid*N + tid] = (uint16_t)w;
}

__global__ void norm_s2(uint16_t *tt){
    /*
     * Normalize -s1 elements into the [-q/2..q/2] range.
     */
    uint32_t tid = threadIdx.x, bid = blockIdx.x;    
    int32_t w;

    w = (int32_t)tt[bid*N + tid];
    w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    ((int16_t *)tt)[bid*N + tid] = (int16_t)w;
}
    // for (u = 0; u < n; u ++) {
    //     int32_t w;

    //     w = (int32_t)tt[u];
    //     w -= (int32_t)(Q & -(((Q >> 1) - (uint32_t)w) >> 31));
    //     ((int16_t *)tt)[u] = (int16_t)w;
    // }

   /*
     * Signature is valid if and only if the aggregate (-s1,s2) vector
     * is short enough.
     */
__global__ void is_short_gpu(int16_t *s1, int16_t *s2, uint32_t *s)
{
    /*
     * We use the l2-norm. Code below uses only 32-bit operations to
     * compute the square of the norm with saturation to 2^32-1 if
     * the value exceeds 2^31-1.
     */
    size_t u;
    uint32_t ng, tmp;
    uint32_t bid = blockIdx.x;       
    tmp = 0;
    ng = 0;
    for (u = 0; u < N; u ++) {
        int32_t z;

        z = s1[bid*N + u];
        tmp += (uint32_t)(z * z);
        ng |= tmp;
        z = s2[bid*N +u];
        tmp += (uint32_t)(z * z);
        ng |= tmp;
    }
    tmp |= -(ng >> 31);
    // printf("%u\n", tmp);
    // s[bid] = tmp;
    if(tmp <= l2bound[9])   //logn=9
    {
        s[bid]=1;            
    }

    if(!s[bid])
        printf("short detected %u\n", s[bid]);
    
}

__global__ void is_short_half_gpu(uint32_t *sqn, const int16_t *s2, uint32_t *s)
{
    size_t n, u;
    uint32_t ng;
    uint32_t bid = blockIdx.x;       

    n = (size_t)1 << 9;
    ng = -(sqn[bid] >> 31);
    for (u = 0; u < n; u ++) {
        int32_t z;

        z = s2[bid*N +u];
        sqn[bid] += (uint32_t)(z * z);
        ng |= sqn[bid];
    }
    sqn[bid] |= -(ng >> 31);
    
    if(sqn[bid] <= l2bound[9])  //logn=9
    {
        s[bid]=1;            
    }
    if(!s[bid])
        printf("short half detected %u\n", s[bid]);     
}


    
// }

// wklee, each thread takes 7 inputs and produce 4 outputs.
__global__ void modq_decode_gpu(uint16_t *x, unsigned logn, uint8_t *in, size_t max_in_len)
{
    uint32_t n, in_len, u, i=0;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    uint32_t acc;
    int acc_len;

    n = (size_t)1 << logn;
    in_len = ((n * 14) + 7) >> 3;
    if (in_len > max_in_len) {
        return ;
    }
    
    acc = 0;
    acc_len = 0;
    u = 0;
    while (u < 4) {        
        // acc = (acc << 8) | (*in ++);
        acc = (acc << 8) | (in[bid*CRYPTO_PUBLICKEYBYTES + tid*7 +i]);
        acc_len += 8;
        if (acc_len >= 14) {
            unsigned w;

            acc_len -= 14;
            w = (acc >> acc_len) & 0x3FFF;
            if (w >= 12289) {
                return ;
            }            
            x[bid*N + tid*4 + u] = (uint16_t)w;
            u ++;
        }        
        i++;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        return ;
    }
    // return in_len;
}

__global__ void msg_len_gpu(uint8_t *sm, uint32_t *msg_len, uint32_t *smlen, uint32_t *sig_len)
{
    uint32_t bid = blockIdx.x;
    uint32_t mlen = MLEN; // make mlen variable in future

    sig_len[bid] = ((uint64_t)sm[bid*(mlen + CRYPTO_BYTES) + 0] << 8) | (uint64_t)sm[bid*(mlen + CRYPTO_BYTES) + 1];
    // if (sig_len > (smlen - 2 - NONCELEN)) {
    //     return -1;
    // }
    msg_len[bid] = smlen[bid] - 2 - NONCELEN - sig_len[bid];
    // printf("%u %u \n", bid, msg_len[bid]);
}


__global__ void comp_decode_gpu(int16_t *x, unsigned logn, uint8_t *in, uint32_t *in_len, uint32_t *msg_len)
{
    uint32_t bid = blockIdx.x;
    uint32_t u, v;
    uint8_t *buf;
    uint32_t acc;
    unsigned acc_len;

    uint32_t max_in_len = in_len[bid] - 1;
    // esig = sm + 2 + NONCELEN + msg_len;
    buf = in + 3 + NONCELEN + msg_len[bid];
    
    acc = 0;
    acc_len = 0;
    v = 0;
    for (u = 0; u < N; u ++) {
        unsigned b, s, m;

        /*
         * Get next eight bits: sign and low seven bits of the
         * absolute value.
         */
        if (v >= max_in_len) {
            return;
        }
        acc = (acc << 8) | (uint32_t)buf[bid*(MLEN+CRYPTO_BYTES) + v ++];
        // printf("%u \n", v);
        b = acc >> acc_len;
        s = b & 128;
        m = b & 127;

        /*
         * Get next bits until a 1 is reached.
         */
        for (;;) {
            if (acc_len == 0) {
                if (v >= max_in_len) {
                    return;
                }
                acc = (acc << 8) | (uint32_t)buf[bid*(MLEN+CRYPTO_BYTES) + v ++];
                acc_len = 8;
                // printf("++%u \n", v);
            }
            acc_len --;
            if (((acc >> acc_len) & 1) != 0) {
                break;
            }
            m += 128;
            if (m > 2047) {
                return;
            }
        }

        /*
         * "-0" is forbidden.
         */
        if (s && m == 0) {
            return;
        }

        x[bid*N + u] = (int16_t)(s ? -(int)m : (int)m);
    }

    /*
     * Unused bits in the last byte must be zero.
     */
    if ((acc & ((1u << acc_len) - 1u)) != 0) {
        return;
    }

    // return v;
}


__global__ void comp_encode_gpu(uint8_t *buf, size_t max_out_len, const int16_t *x, unsigned logn, uint32_t *len)
{
    // uint8_t *buf;
    size_t n, u, v;
    uint32_t acc;
    unsigned acc_len;
    uint32_t bid = blockIdx.x;
    uint32_t outlen = (CRYPTO_BYTES - 2 - NONCELEN);

    n = (size_t)1 << logn;
    // buf = out;

    /*
     * Make sure that all values are within the -2047..+2047 range.
     */
    for (u = 0; u < n; u ++) {
        if (x[bid*N + u] < -2047 || x[bid*N + u] > +2047) {
            return ;
        }
    }

    acc = 0;
    acc_len = 0;
    v = 0;
    for (u = 0; u < n; u ++) {
        int t;
        unsigned w;

        /*
         * Get sign and absolute value of next integer; push the
         * sign bit.
         */
        acc <<= 1;
        t = x[bid*N + u];
        if (t < 0) {
            t = -t;
            acc |= 1;
        }
        w = (unsigned)t;

        /*
         * Push the low 7 bits of the absolute value.
         */
        acc <<= 7;
        acc |= w & 127u;
        w >>= 7;

        /*
         * We pushed exactly 8 bits.
         */
        acc_len += 8;

        /*
         * Push as many zeros as necessary, then a one. Since the
         * absolute value is at most 2047, w can only range up to
         * 15 at this point, thus we will add at most 16 bits
         * here. With the 8 bits above and possibly up to 7 bits
         * from previous iterations, we may go up to 31 bits, which
         * will fit in the accumulator, which is an uint32_t.
         */
        acc <<= (w + 1);
        acc |= 1;
        acc_len += w + 1;

        /*
         * Produce all full bytes.
         */
        while (acc_len >= 8) {
            acc_len -= 8;
            if (buf != NULL) {
                if (v >= max_out_len) {
                    return;
                }
                buf[bid*outlen + v] = (uint8_t)(acc >> acc_len);
            }
            v ++;
        }
    }

    /*
     * Flush remaining bits (if any).
     */
    if (acc_len > 0) {
        if (buf != NULL) {
            if (v >= max_out_len) {
                return ;
            }
            buf[bid*outlen + v] = (uint8_t)(acc << (8 - acc_len));
        }
        v ++;
    }

    len[bid] = v+1;
    // return v;
}

__global__ void write_smlen_gpu(uint8_t *sm, uint32_t *sig_len)
{
    uint32_t bid = blockIdx.x;    
    // printf("%u\n", sig_len[bid]);
    sm[bid*(MLEN+CRYPTO_BYTES) + 0] = (unsigned char)(sig_len[bid] >> 8);
    sm[bid*(MLEN+CRYPTO_BYTES) + 1] = (unsigned char)sig_len[bid];
}

__global__ void byte_cmp(uint8_t *m, uint8_t *m1)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;        

    if(m[bid*MLEN + tid] != m1[bid*(MLEN+CRYPTO_BYTES) + tid]){
        printf("wrong signature at %u %u: %u %u \n", bid, tid, m[bid*MLEN + tid], m1[bid*(MLEN+CRYPTO_BYTES) + tid]);
        
    }
}


/*
 * Division by 2 modulo q. Operand must be in the 0..q-1 range.
 */
__device__ static inline uint32_t mq_rshift1(uint32_t x)
{
    x += Q & -(x & 1);
    return (x >> 1);
}

/*
 * Montgomery multiplication modulo q. If we set R = 2^16 mod q, then
 * this function computes: x * y / R mod q
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
 * Montgomery squaring (computes (x^2)/R).
 */
__device__ static inline uint32_t mq_montysqr(uint32_t x)
{
    return mq_montymul(x, x);
}

/*
 * Divide x by y modulo q = 12289.
 */
__device__ static inline uint32_t mq_div_12289(uint32_t x, uint32_t y)
{
    /*
     * We invert y by computing y^(q-2) mod q.
     *
     * We use the following addition chain for exponent e = 12287:
     *
     *   e0 = 1
     *   e1 = 2 * e0 = 2
     *   e2 = e1 + e0 = 3
     *   e3 = e2 + e1 = 5
     *   e4 = 2 * e3 = 10
     *   e5 = 2 * e4 = 20
     *   e6 = 2 * e5 = 40
     *   e7 = 2 * e6 = 80
     *   e8 = 2 * e7 = 160
     *   e9 = e8 + e2 = 163
     *   e10 = e9 + e8 = 323
     *   e11 = 2 * e10 = 646
     *   e12 = 2 * e11 = 1292
     *   e13 = e12 + e9 = 1455
     *   e14 = 2 * e13 = 2910
     *   e15 = 2 * e14 = 5820
     *   e16 = e15 + e10 = 6143
     *   e17 = 2 * e16 = 12286
     *   e18 = e17 + e0 = 12287
     *
     * Additions on exponents are converted to Montgomery
     * multiplications. We define all intermediate results as so
     * many local variables, and let the C compiler work out which
     * must be kept around.
     */
    uint32_t y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;
    uint32_t y10, y11, y12, y13, y14, y15, y16, y17, y18;

    y0 = mq_montymul(y, R2);
    y1 = mq_montysqr(y0);
    y2 = mq_montymul(y1, y0);
    y3 = mq_montymul(y2, y1);
    y4 = mq_montysqr(y3);
    y5 = mq_montysqr(y4);
    y6 = mq_montysqr(y5);
    y7 = mq_montysqr(y6);
    y8 = mq_montysqr(y7);
    y9 = mq_montymul(y8, y2);
    y10 = mq_montymul(y9, y8);
    y11 = mq_montysqr(y10);
    y12 = mq_montysqr(y11);
    y13 = mq_montymul(y12, y9);
    y14 = mq_montysqr(y13);
    y15 = mq_montysqr(y14);
    y16 = mq_montymul(y15, y10);
    y17 = mq_montysqr(y16);
    y18 = mq_montymul(y17, y0);

    /*
     * Final multiplication with x, which is not in Montgomery
     * representation, computes the correct division result.
     */
    return mq_montymul(y18, x);
}


/*
 * Reduce a small signed integer modulo q. The source integer MUST
 * be between -q/2 and +q/2.
 */
__device__ static inline uint32_t mq_conv_small(int x)
{
    /*
     * If x < 0, the cast to uint32_t will set the high bit to 1.
     */
    uint32_t y;

    y = (uint32_t)x;
    y += Q & -(y >> 31);
    return y;
}

/*
 * Reduce a small signed integer modulo q. The source integer MUST
 * be between -q/2 and +q/2.
 */
__global__ void mq_conv_small_gpu(uint16_t *d, int8_t *x)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    d[bid*N + tid] = (uint16_t)mq_conv_small(x[bid*N + tid]);
}

__global__ void complete_private_gpu(int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F,
    unsigned logn, uint16_t *t1, uint16_t *t2)
{
    size_t u, n;
    // uint16_t *t1, *t2;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    n = (size_t)1 << logn;
    // t1 = (uint16_t *)tmp;
    // t2 = t1 + n;

    t1[bid*N + tid] = (uint16_t)mq_conv_small(g[bid*N + tid]);
    t2[bid*N + tid] = (uint16_t)mq_conv_small(F[bid*N + tid]);
    // t1[tid+N/2] = (uint16_t)mq_conv_small(g[tid+N/2]);
    // t2[tid+N/2] = (uint16_t)mq_conv_small(F[tid+N/2]);    
}

__global__ void complete_private_gpu2(uint16_t *t1, uint16_t *t2)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    if (t2[bid*N + tid] == 0) {
        return;
    }
    t1[bid*N + tid] = (uint16_t)mq_div_12289(t1[bid*N + tid], t2[bid*N + tid]);    
}

__global__ void complete_private_gpu3(uint16_t *t1, int8_t *G)
{
    uint32_t tid = threadIdx.x, bid = blockIdx.x;
    
    uint32_t w;
    int32_t gi;

    w = t1[bid*N + tid];
    w -= (Q & ~-((w - (Q >> 1)) >> 31));
    gi = *(int32_t *)&w;
    if (gi < -127 || gi > +127) {
        return;
    }
    G[bid*N + tid] = (int8_t)gi;
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

__global__ void complete_private_comb_gpu(int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F,
    unsigned logn, uint16_t *t1, uint16_t *t2)
{
    size_t u, n;    
    size_t t, m;
    uint32_t tid = threadIdx.x, bid = blockIdx.x;

    n = (size_t)1 << logn;    

    t1[bid*N + tid] = (uint16_t)mq_conv_small(g[bid*N + tid]);
    t2[bid*N + tid] = (uint16_t)mq_conv_small(F[bid*N + tid]);
    t1[bid*N + tid+N/2] = (uint16_t)mq_conv_small(g[bid*N + tid+N/2]);
    t2[bid*N + tid+N/2] = (uint16_t)mq_conv_small(F[bid*N + tid+N/2]);    

    // mq_NTT
    __shared__ uint16_t s_a[N];
    s_a[tid] = t1[bid*N + tid];
    s_a[tid+256] = t1[bid*N + tid+256];
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

    t1[bid*N + tid] = s_a[tid];
    t1[bid*N + tid+256] = s_a[tid+256];

    s_a[tid] = t2[bid*N + tid];
    s_a[tid+256] = t2[bid*N + tid+256];
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

    t2[bid*N + tid] = s_a[tid];
    t2[bid*N + tid+256] = s_a[tid+256];
    // mq_poly_tomonty
    t1[bid*N + tid] = (uint16_t)mq_montymul(t1[bid*N + tid], R2);
    t1[bid*N + N/2 + tid] = (uint16_t)mq_montymul(t1[bid*N + N/2 + tid], R2); 
    // mq_poly_montymul_ntt_gpu
    t1[bid*N + tid] = (uint16_t)mq_montymul(t1[bid*N + tid], t2[bid*N + tid]);
    t1[bid*N + N/2 + tid] = (uint16_t)mq_montymul(t1[bid*N  + N/2 + tid], t2[bid*N + N/2 + tid]);

    t2[bid*N + tid] = (uint16_t)mq_conv_small(f[bid*N + tid]);
    t2[bid*N + N/2 + tid] = (uint16_t)mq_conv_small(f[bid*N + N/2 + tid]);
    // mq_NTT
    s_a[tid] = t2[bid*N + tid];
    s_a[tid+256] = t2[bid*N + tid+256];
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
    // t2[bid*N + tid] = s_a[tid];
    // t2[bid*N + tid+256] = s_a[tid+256];
    
    // if (t2[bid*N + tid] == 0) {
    //     return;
    // }
    // t1[bid*N + tid] = (uint16_t)mq_div_12289(t1[bid*N + tid], t2[bid*N + tid]);    
    // if (t2[bid*N + N/2 + tid] == 0) {
    //     return;
    // }
    // t1[bid*N + N/2 + tid] = (uint16_t)mq_div_12289(t1[bid*N + N/2 + tid], t2[bid*N + N/2 + tid]);  

    if (s_a[tid] == 0) {
        return;
    }
    t1[bid*N + tid] = (uint16_t)mq_div_12289(t1[bid*N + tid], s_a[tid]);    
    if (s_a[N/2 + tid] == 0) {
        return;
    }
    t1[bid*N + N/2 + tid] = (uint16_t)mq_div_12289(t1[bid*N + N/2 + tid], s_a[N/2 + tid]);  

    // mq_iNTT
    uint32_t ni = 128;  // pre-computed from falcon512fpu
    uint32_t v, w;
    uint32_t hm, dt;
    
    t = 1;
    s_a[tid] = t1[bid*N + tid];
    s_a[tid+256] = t1[bid*N + tid+256];
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
    
    // t1[bid*N + tid] = (uint16_t)mq_montymul(s_a[tid], ni);
    // t1[bid*N + N/2 + tid] = (uint16_t)mq_montymul(s_a[N/2 + tid], ni);       
    s_a[tid] = (uint16_t)mq_montymul(s_a[tid], ni);
    s_a [N/2 + tid] = (uint16_t)mq_montymul(s_a[N/2 + tid], ni);  
    int32_t gi;

    w = s_a [tid];
    w -= (Q & ~-((w - (Q >> 1)) >> 31));
    gi = *(int32_t *)&w;
    if (gi < -127 || gi > +127) {
        return;
    }
    G[bid*N + tid] = (int8_t)gi;   

    w = s_a[N/2 + tid];
    w -= (Q & ~-((w - (Q >> 1)) >> 31));
    gi = *(int32_t *)&w;
    if (gi < -127 || gi > +127) {
        return;
    }
    G[bid*N + N/2 + tid] = (int8_t)gi;    
}

__global__ void trim_i8_decode_gpu(int8_t *x, int8_t *y, int8_t *z, unsigned logn, unsigned bits, uint8_t *buf, size_t max_in_len)
{
    size_t n, in_len, u;
    uint32_t acc, mask1, mask2, i, count = 0, bid = blockIdx.x;
    unsigned acc_len;

    n = (size_t)1 << logn;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0;
    acc = 0;
    acc_len = 0;
    mask1 = ((uint32_t)1 << bits) - 1;
    mask2 = (uint32_t)1 << (bits - 1);
    while (u < n) {
        
        acc = (acc << 8) | buf[count + bid*CRYPTO_SECRETKEYBYTES];
        // printf("%u %u\n", bid, buf[count + bid*CRYPTO_SECRETKEYBYTES]);
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            x[bid*N + u] = (int8_t)*(int32_t *)&w;
            u += 1;
        }
        count = count+1;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }
    
    max_in_len-= in_len;
    buf+= in_len;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0;  count = 0;
    acc = 0;
    acc_len = 0;
    while (u < n) {
        acc = (acc << 8) | buf[count + bid*CRYPTO_SECRETKEYBYTES];
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            y[bid*N + u] = (int8_t)*(int32_t *)&w;
            u += 1;

        }
        count = count+1;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }

    bits = 8;   //max_FG_bits
    max_in_len-= in_len;
    buf+= in_len;
    in_len = ((n * bits) + 7) >> 3;
    if (in_len > max_in_len) {
        return;
    }

    u = 0; count = 0;
    acc = 0;
    acc_len = 0;
    mask1 = ((uint32_t)1 << bits) - 1;
    mask2 = (uint32_t)1 << (bits - 1);
    while (u < n) {
        acc = (acc << 8) | buf[count + bid*CRYPTO_SECRETKEYBYTES];
        acc_len += 8;
        while (acc_len >= bits && u < n) {
            uint32_t w;

            acc_len -= bits;
            w = (acc >> acc_len) & mask1;
            w |= -(w & mask2);
            if (w == -mask2) {
                /*
                 * The -2^(bits-1) value is forbidden.
                 */
                return ;
            }
            z[bid*N + u] = (int8_t)*(int32_t *)&w;
            u += 1;

        }
        count = count+1;
    }
    if ((acc & (((uint32_t)1 << acc_len) - 1)) != 0) {
        /*
         * Extra bits in the last byte must be zero.
         */
        return ;
    }

}