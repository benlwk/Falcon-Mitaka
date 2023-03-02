#include <stdio.h>
#include <stdint.h>
#include "fpr.cuh"
#include "cuda_kernel.cuh"

__global__ void check2(int16_t* s2tmp, fpr *t1);
__global__ void check1(int16_t* s1tmp, fpr *t0, uint16_t *hm, uint32_t *sqn);
     /*  
     * Reduce s2 elements modulo q ([0..q-1] range).
     */
__global__ void reduce_s2(int16_t *s2, uint16_t *tt);

__global__ void norm_s2(uint16_t *tt);
/*
 * Acceptance bound for the (squared) l2-norm of the signature depends
 * on the degree. This array is indexed by logn (1 to 10). These bounds
 * are _inclusive_ (they are equal to floor(beta^2)).
 */
__device__ static const uint32_t l2bound[] = {
    0,    /* unused */
    101498,
    208714,
    428865,
    892039,
    1852696,
    3842630,
    7959734,
    16468416,
    34034726,
    70265242
};


/*
 * Key elements and signatures are polynomials with small integer
 * coefficients. Here are some statistics gathered over many
 * generated key pairs (10000 or more for each degree):
 *
 *   log(n)     n   max(f,g)   std(f,g)   max(F,G)   std(F,G)
 *      1       2     129       56.31       143       60.02
 *      2       4     123       40.93       160       46.52
 *      3       8      97       28.97       159       38.01
 *      4      16     100       21.48       154       32.50
 *      5      32      71       15.41       151       29.36
 *      6      64      59       11.07       138       27.77
 *      7     128      39        7.91       144       27.00
 *      8     256      32        5.63       148       26.61
 *      9     512      22        4.00       137       26.46
 *     10    1024      15        2.84       146       26.41
 *
 * We want a compact storage format for private key, and, as part of
 * key generation, we are allowed to reject some keys which would
 * otherwise be fine (this does not induce any noticeable vulnerability
 * as long as we reject only a small proportion of possible keys).
 * Hence, we enforce at key generation time maximum values for the
 * elements of f, g, F and G, so that their encoding can be expressed
 * in fixed-width values. Limits have been chosen so that generated
 * keys are almost always within bounds, thus not impacting neither
 * security or performance.
 *
 * IMPORTANT: the code assumes that all coefficients of f, g, F and G
 * ultimately fit in the -127..+127 range. Thus, none of the elements
 * of max_fg_bits[] and max_FG_bits[] shall be greater than 8.
 */

static uint8_t max_fg_bits[] = {
    0, /* unused */
    8,
    8,
    8,
    8,
    8,
    7,
    7,
    6,
    6,
    5
};

static uint8_t max_FG_bits[] = {
    0, /* unused */
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8
};

/*
 * When generating a new key pair, we can always reject keys which
 * feature an abnormally large coefficient. This can also be done for
 * signatures, albeit with some care: in case the signature process is
 * used in a derandomized setup (explicitly seeded with the message and
 * private key), we have to follow the specification faithfully, and the
 * specification only enforces a limit on the L2 norm of the signature
 * vector. The limit on the L2 norm implies that the absolute value of
 * a coefficient of the signature cannot be more than the following:
 *
 *   log(n)     n   max sig coeff (theoretical)
 *      1       2       412
 *      2       4       583
 *      3       8       824
 *      4      16      1166
 *      5      32      1649
 *      6      64      2332
 *      7     128      3299
 *      8     256      4665
 *      9     512      6598
 *     10    1024      9331
 *
 * However, the largest observed signature coefficients during our
 * experiments was 1077 (in absolute value), hence we can assume that,
 * with overwhelming probability, signature coefficients will fit
 * in -2047..2047, i.e. 12 bits.
 */

static uint8_t max_sig_bits[] = {
    0, /* unused */
    10,
    11,
    11,
    12,
    12,
    12,
    12,
    12,
    12,
    12
};


    /*
     * Signature is valid if and only if the aggregate (-s1,s2) vector
     * is short enough.
     */
__global__ void is_short_gpu(int16_t *s1, int16_t *s2, uint32_t *s);
__global__ void is_short_half_gpu(uint32_t *sqn, const int16_t *s2,  uint32_t *s);
__global__ void modq_decode_gpu(uint16_t *x, unsigned logn, uint8_t *in, size_t max_in_len);
__global__ void msg_len_gpu(uint8_t *sm, uint32_t *msg_len, uint32_t *smlen, uint32_t *sig_len);
__global__ void comp_decode_gpu(int16_t *x, unsigned logn, uint8_t *in, uint32_t *in_len, uint32_t *msg_len);
__global__ void comp_encode_gpu(uint8_t *out, size_t max_out_len, const int16_t *x, unsigned logn, uint32_t *len);
__global__ void write_smlen_gpu(uint8_t *sm, uint32_t *sig_len);
__global__ void byte_cmp(uint8_t *m, uint8_t *m1);
__global__ void complete_private_gpu(int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F,
    unsigned logn, uint16_t *t1, uint16_t *t2);
__global__ void mq_conv_small_gpu(uint16_t *d, int8_t *x);
__global__ void complete_private_gpu2(uint16_t *t1, uint16_t *t2);
__global__ void complete_private_gpu3(uint16_t *t1, int8_t *G);
__global__ void complete_private_comb_gpu(int8_t *G, const int8_t *f, const int8_t *g, const int8_t *F,
    unsigned logn, uint16_t *t1, uint16_t *t2);
__global__ void trim_i8_decode_gpu(int8_t *x, int8_t *y, int8_t *z, unsigned logn, unsigned bits, uint8_t *buf, size_t max_in_len);