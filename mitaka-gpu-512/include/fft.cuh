#include <stdio.h>
#include <stdint.h>
#include "fpr.cuh"
#include "cuda_kernel.cuh"


// #pragma once
#ifndef FFT_CUH__
#define FFT_CUH__
/*
 * Addition of two complex numbers (d = a + b).
 */
#define FPC_ADD(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_add(a_re, b_re); \
        fpct_im = fpr_add(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Subtraction of two complex numbers (d = a - b).
 */
#define FPC_SUB(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_sub(a_re, b_re); \
        fpct_im = fpr_sub(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Multplication of two complex numbers (d = a * b).
 */
#define FPC_MUL(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_b_re, fpct_b_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_b_re = (b_re); \
        fpct_b_im = (b_im); \
        fpct_d_re = fpr_sub( \
            fpr_mul(fpct_a_re, fpct_b_re), \
            fpr_mul(fpct_a_im, fpct_b_im)); \
        fpct_d_im = fpr_add( \
            fpr_mul(fpct_a_re, fpct_b_im), \
            fpr_mul(fpct_a_im, fpct_b_re)); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)

/*
 * Squaring of a complex number (d = a * a).
 */
#define FPC_SQR(d_re, d_im, a_re, a_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_d_re = fpr_sub(fpr_sqr(fpct_a_re), fpr_sqr(fpct_a_im)); \
		fpct_d_im = fpr_double(fpr_mul(fpct_a_re, fpct_a_im)); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)

/*
 * Inversion of a complex number (d = 1 / a).
 */
#define FPC_INV(d_re, d_im, a_re, a_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpr fpct_m; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_m = fpr_add(fpr_sqr(fpct_a_re), fpr_sqr(fpct_a_im)); \
		fpct_m = fpr_inv(fpct_m); \
		fpct_d_re = fpr_mul(fpct_a_re, fpct_m); \
		fpct_d_im = fpr_mul(fpr_neg(fpct_a_im), fpct_m); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)

/*
 * Division of complex numbers (d = a / b).
 */
#define FPC_DIV(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_b_re, fpct_b_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpr fpct_m; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_b_re = (b_re); \
		fpct_b_im = (b_im); \
		fpct_m = fpr_add(fpr_sqr(fpct_b_re), fpr_sqr(fpct_b_im)); \
		fpct_m = fpr_inv(fpct_m); \
		fpct_b_re = fpr_mul(fpct_b_re, fpct_m); \
		fpct_b_im = fpr_mul(fpr_neg(fpct_b_im), fpct_m); \
		fpct_d_re = fpr_sub( \
			fpr_mul(fpct_a_re, fpct_b_re), \
			fpr_mul(fpct_a_im, fpct_b_im)); \
		fpct_d_im = fpr_add( \
			fpr_mul(fpct_a_re, fpct_b_im), \
			fpr_mul(fpct_a_im, fpct_b_re)); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)

// wklee these are for CPU only
/*
 * Addition of two complex numbers (d = a + b).
 */
#define FPC_ADD2(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_add2(a_re, b_re); \
        fpct_im = fpr_add2(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Subtraction of two complex numbers (d = a - b).
 */
#define FPC_SUB2(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_sub2(a_re, b_re); \
        fpct_im = fpr_sub2(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Multplication of two complex numbers (d = a * b).
 */
#define FPC_MUL2(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_b_re, fpct_b_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_b_re = (b_re); \
        fpct_b_im = (b_im); \
        fpct_d_re = fpr_sub2( \
            fpr_mul2(fpct_a_re, fpct_b_re), \
            fpr_mul2(fpct_a_im, fpct_b_im)); \
        fpct_d_im = fpr_add2( \
            fpr_mul2(fpct_a_re, fpct_b_im), \
            fpr_mul2(fpct_a_im, fpct_b_re)); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)

/*
 * Squaring of a complex number (d = a * a).
 */
#define FPC_SQR(d_re, d_im, a_re, a_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_d_re = fpr_sub(fpr_sqr(fpct_a_re), fpr_sqr(fpct_a_im)); \
		fpct_d_im = fpr_double(fpr_mul(fpct_a_re, fpct_a_im)); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)

/*
 * Inversion of a complex number (d = 1 / a).
 */
#define FPC_INV2(d_re, d_im, a_re, a_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpr fpct_m; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_m = fpr_add2(fpr_sqr2(fpct_a_re), fpr_sqr2(fpct_a_im)); \
		fpct_m = fpr_inv2(fpct_m); \
		fpct_d_re = fpr_mul2(fpct_a_re, fpct_m); \
		fpct_d_im = fpr_mul2(fpr_neg2(fpct_a_im), fpct_m); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)

/*
 * Division of complex numbers (d = a / b).
 */
#define FPC_DIV2(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
		fpr fpct_a_re, fpct_a_im; \
		fpr fpct_b_re, fpct_b_im; \
		fpr fpct_d_re, fpct_d_im; \
		fpr fpct_m; \
		fpct_a_re = (a_re); \
		fpct_a_im = (a_im); \
		fpct_b_re = (b_re); \
		fpct_b_im = (b_im); \
		fpct_m = fpr_add2(fpr_sqr2(fpct_b_re), fpr_sqr2(fpct_b_im)); \
		fpct_m = fpr_inv2(fpct_m); \
		fpct_b_re = fpr_mul2(fpct_b_re, fpct_m); \
		fpct_b_im = fpr_mul2(fpr_neg2(fpct_b_im), fpct_m); \
		fpct_d_re = fpr_sub2( \
			fpr_mul2(fpct_a_re, fpct_b_re), \
			fpr_mul2(fpct_a_im, fpct_b_im)); \
		fpct_d_im = fpr_add2( \
			fpr_mul2(fpct_a_re, fpct_b_im), \
			fpr_mul2(fpct_a_im, fpct_b_re)); \
		(d_re) = fpct_d_re; \
		(d_im) = fpct_d_im; \
	} while (0)	

void poly_LDL_fft_s(const fpr *g00, fpr *g01, fpr *g11, unsigned logn);
void poly_split_fft_s(fpr *f0, fpr *f1, const fpr *f, unsigned logn);
void poly_merge_fft_s(fpr *f, const fpr *f0, const fpr *f1, unsigned logn);
void poly_add_s(fpr *a, const fpr *b, unsigned logn);
void poly_sub_s(fpr * a, const fpr * b, unsigned logn);
void poly_mul_fft_s(fpr *a, const fpr *b, unsigned logn);



__constant__ uint32_t S [MITAKA_D]= {10810, 7143, 4043, 10984, 722, 5736, 8155, 3542, 8785, 9744, 3621, 10643, 1212, 3195, 5860, 7468, 2639, 9664, 11340, 11726, 9314, 9283, 9545, 5728, 7698, 5023, 5828, 8961, 6512, 7311, 1351, 2319, 11119, 11334, 11499, 9088, 3014, 5086, 10963, 4846, 9542, 9154, 3712, 4805, 8736, 11227, 9995, 3091, 12208, 7969, 11289, 9326, 7393, 9238, 2366, 11112, 8034, 10654, 9521, 12149, 10436, 7678, 11563, 1260, 4388, 4632, 6534, 2426, 334, 1428, 1696, 2013, 9000, 729, 3241, 2881, 3284, 7197, 10200, 8595, 7110, 10530, 8582, 3382, 11934, 9741, 8058, 3637, 3459, 145, 6747, 9558, 8357, 7399, 6378, 9447, 480, 1022,  9, 9821, 339, 5791, 544, 10616, 4278, 6958, 7300, 8112, 8705, 1381, 9764, 11336, 8541, 827, 5767, 2476, 118, 2197, 7222, 3949, 8993, 4452, 2396, 7935, 130, 2837, 6915, 2401, 442, 7188, 11222, 390, 773, 8456, 3778, 354, 4861, 9377, 5698, 5012, 9808, 2859, 11244, 1017, 7404, 1632, 7205, 27, 9223, 8526, 10849, 1537, 242, 4714, 8146, 9611, 3704, 5019, 11744, 1002, 5011, 5088, 8005, 7313, 10682, 8509, 11414, 9852, 3646, 6022, 2987, 9723, 10102, 6250, 9867, 11224, 2143, 11885, 7644, 1168, 5277, 11082, 3248, 493, 8193, 6845, 2381, 7952, 11854, 1378, 1912, 2166, 3915, 12176, 7370, 12129, 3149, 12286, 4437, 3636, 4938, 5291, 2704, 10863, 7635, 1663, 10512, 3364, 1689, 4057, 9018, 9442, 7875, 2174, 4372, 7247, 9984, 4053, 2645, 5195, 9509, 7394, 1484, 9042, 9603, 8311, 9320, 9919, 2865, 5332, 3510, 1630, 10163, 5407, 3186, 11136, 9405, 10040, 8241, 9890, 8889, 7098, 9153, 9289, 671, 3016, 243, 6730, 420, 10111, 1544, 3985, 4905, 3531, 476, 49, 1263, 5915, 1483, 9789, 10800, 10706, 6347, 1512, 350, 10474, 5383, 5369, 10232, 9087, 4493, 9551, 6421, 6554, 2655, 9280, 1693, 174, 723, 10314, 8532, 347, 2925, 8974, 11863, 1858, 4754, 3030, 4115, 2361, 10446, 2908, 218, 3434, 8760, 3963, 576, 6142, 9842, 1954, 10238, 9407, 10484, 3991, 8320, 9522, 156, 2281, 5876, 10258, 5333, 3772, 418, 5908, 11836, 5429, 7515, 7552, 1293, 295, 6099, 5766, 652, 8273, 4077, 8527, 9370, 325, 10885, 11143, 11341, 5990, 1159, 8561, 8240, 3329, 4298, 12121, 2692, 5961, 7183, 10327, 1594, 6167, 9734, 7105, 11089, 1360, 3956, 6170, 5297, 8210, 11231, 922, 441, 1958, 4322, 1112, 2078, 4046, 709, 9139, 1319, 4240, 8719, 6224, 11454, 2459, 683, 3656, 12225, 10723, 5782, 9341, 9786, 9166, 10542, 9235, 6803, 7856, 6370, 3834, 7032, 7048, 9369, 8120, 9162, 6821, 1010, 8807, 787, 5057, 4698, 4780, 8844, 12097, 1321, 4912, 10240, 677, 6415, 6234, 8953, 1323, 9523, 12237, 3174, 1579, 11858, 9784, 5906, 3957, 9450, 151, 10162, 12231, 12048, 3532, 11286, 1956, 7280, 11404, 6281, 3477, 6608, 142, 11184, 9445, 3438, 11314, 4212, 9260, 6695, 4782, 5886, 8076, 504, 2302, 11684, 11868, 8209, 3602, 6068, 8689, 3263, 6077, 7665, 7822, 7500, 6752, 4749, 4449, 6833, 12142, 8500, 6118, 8471, 1190, 9606, 3860, 5445, 7753, 11239, 5079, 9027, 2169, 11767, 7965, 4916, 8214, 5315, 11011, 9945, 1973, 6715, 8775, 11248, 5925, 11271, 654, 3565, 1702, 1987, 6760, 5206, 3199, 12233, 6136, 6427, 6874, 8646, 4948, 6152, 400, 10561, 5339, 5446, 3710, 6093, 468, 8301, 316, 11907, 10256, 8291, 3879, 1922, 10930, 6854, 973, 11035};

__constant__ uint32_t inv_S [MITAKA_D]= {1254, 11316, 5435, 1359, 10367, 8410, 3998, 2033, 382, 11973, 3988, 11821, 6196, 8579, 6843, 6950, 1728, 11889, 6137, 7341, 3643, 5415, 5862, 6153, 56, 9090, 7083, 5529, 10302, 10587, 8724, 11635, 1018, 6364, 1041, 3514, 5574, 10316, 2344, 1278, 6974, 4075, 7373, 4324, 522, 10120, 3262, 7210, 1050, 4536, 6844, 8429, 2683, 11099, 3818, 6171, 3789, 147, 5456, 7840, 7540, 5537, 4789, 4467, 4624, 6212, 9026, 3600, 6221, 8687, 4080, 421, 605, 9987, 11785, 4213, 6403, 7507, 5594, 3029, 8077, 975, 8851, 2844, 1105, 12147, 5681, 8812, 6008, 885, 5009, 10333, 1003, 8757, 241, 58, 2127, 12138, 2839, 8332, 6383, 2505, 431, 10710, 9115, 52, 2766, 10966, 3336, 6055, 5874, 11612, 2049, 7377, 10968, 192, 3445, 7509, 7591, 7232, 11502, 3482, 11279, 5468, 3127, 4169, 2920, 5241, 5257, 8455, 5919, 4433, 5486, 3054, 1747, 3123, 2503, 2948, 6507, 1566, 64, 8633, 11606, 9830, 835, 6065, 3570, 8049, 10970, 3150, 11580, 8243, 10211, 11177, 7967, 10331, 11848, 11367, 1058, 4079, 6992, 6119, 8333, 10929, 1200, 5184, 2555, 6122, 10695, 1962, 5106, 6328, 9597, 168, 7991, 8960, 4049, 3728, 11130, 6299, 948, 1146, 1404, 11964, 2919, 3762, 8212, 4016, 11637, 6523, 6190, 11994, 10996, 4737, 4774, 6860, 453, 6381, 11871, 8517, 6956, 2031, 6413, 10008, 12133, 2767, 3969, 8298, 1805, 2882, 2051, 10335, 2447, 6147, 11713, 8326, 3529, 8855, 12071, 9381, 1843, 9928, 8174, 9259, 7535, 10431, 426, 3315, 9364, 11942, 3757, 1975, 11566, 12115, 10596, 3009, 9634, 5735, 5868, 2738, 7796, 3202, 2057, 6920, 6906, 1815, 11939, 10777, 5942, 1583, 1489, 2500, 10806, 6374, 11026, 12240, 11813, 8758, 7384, 8304, 10745, 2178, 11869, 5559, 12046, 9273, 11618, 3000, 3136, 5191, 3400, 2399, 4048, 2249, 2884, 1153, 9103, 6882, 2126, 10659, 8779, 6957, 9424, 2370, 2969, 3978, 2686, 3247, 10805, 4895, 2780, 7094, 9644, 8236, 2305, 5042, 7917, 10115, 4414, 2847, 3271, 8232, 10600, 8925, 1777, 10626, 4654, 1426, 9585, 6998, 7351, 8653, 7852,  3, 9140, 160, 4919, 113, 8374, 10123, 10377, 10911, 435, 4337, 9908, 5444, 4096, 11796, 9041, 1207, 7012, 11121, 4645, 404, 10146, 1065, 2422, 6039, 2187, 2566, 9302, 6267, 8643, 2437, 875, 3780, 1607, 4976, 4284, 7201, 7278, 11287, 545, 7270, 8585, 2678, 4143, 7575, 12047, 10752, 1440, 3763, 3066, 12262, 5084, 10657, 4885, 11272, 1045, 9430, 2481, 7277, 6591, 2912, 7428, 11935, 8511, 3833, 11516, 11899, 1067, 5101, 11847, 9888, 5374, 9452, 12159, 4354, 9893, 7837, 3296, 8340, 5067, 10092, 12171, 9813, 6522, 11462, 3748, 953, 2525, 10908, 3584, 4177, 4989, 5331, 8011, 1673, 11745, 6498, 11950, 2468, 12280, 11267, 11809, 2842, 5911, 4890, 3932, 2731, 5542, 12144, 8830, 8652, 4231, 2548, 355, 8907, 3707, 1759, 5179, 3694, 2089, 5092, 9005, 9408, 9048, 11560, 3289, 10276, 10593, 10861, 11955, 9863, 5755, 7657, 7901, 11029, 726, 4611, 1853, 140, 2768, 1635, 4255, 1177, 9923, 3051, 4896, 2963, 1000, 4320, 81, 9198, 2294, 1062, 3553, 7484, 8577, 3135, 2747, 7443, 1326, 7203, 9275, 3201, 790, 955, 1170, 9970, 10938, 4978, 5777, 3328, 6461, 7266, 4591, 6561, 2744, 3006, 2975, 563, 949, 2625, 9650, 4821, 6429, 9094, 11077, 1646, 8668, 2545, 3504, 8747, 4134, 6553, 11567, 1305, 8246, 5146, 1479};

__global__ void poly_set_g(fpr *out, const int16_t *in);
__global__ void poly_mulselfadj_fft_g(fpr *a);
__global__ void poly_muladj_fft_g(fpr *a, const fpr *b);
__global__ void poly_add_g(fpr *a, const fpr *b);
__global__ void poly_sub_g(fpr *a, const fpr *b);
__global__ void FFT_SM_g(fpr *f, uint32_t in_s, uint32_t out_s);
__global__ void FFT_g(fpr *f, uint32_t in_s, uint32_t out_s);
__global__ void iFFT_g(fpr *f, uint32_t in_s, uint32_t out_s);
__global__ void poly_mul_fft(fpr *a, fpr *b);
__global__ void poly_mulconst(fpr *a, fpr x);
__global__ void smallints_to_fpr_g(fpr *r, int8_t *t, uint32_t in_s, uint32_t out_s);
__global__ void poly_neg_g(fpr *a, uint32_t in_s, uint32_t out_s);
__global__ void poly_copy(fpr *out, fpr *in);
__global__ void poly_copy_u16(int16_t *out, fpr *in);
__global__ void poly_copy_u32(int32_t *out, fpr *in);
__global__ void mq_NTT_gpu(uint16_t *a, unsigned logn);
__global__ void mq_poly_montymul_ntt_gpu(uint16_t *f, uint16_t *g);
__global__ void mq_iNTT_gpu(uint16_t *a);
__global__ void mq_poly_add(uint16_t *f, fpr *g);
__global__ void mq_poly_tomonty(uint16_t *f);
__global__ void comb_all_kernels(uint16_t *a, int16_t *s2, uint16_t *g, uint16_t *h);
__global__ void mq_poly_tomonty(uint16_t *f);
__global__ void reduce_s2(int16_t *s2, uint16_t *tt);
__global__ void norm_s2(uint16_t *tt);
__device__ void poly_LDL_fft(const fpr * g00, fpr * g01, fpr * g11, unsigned logn);
__device__ void poly_split_fft(fpr * f0, fpr *f1,
	const fpr * f, unsigned logn);
__device__ void poly_merge_fft(fpr * f,	const fpr * f0, const fpr * f1, unsigned logn);
__device__ void poly_add(fpr * a, const fpr * b, unsigned logn);
__device__ void poly_sub(fpr * a, const fpr * b, unsigned logn);
__global__ void poly_recenter(fpr* p);
__global__ void check_norm(fpr* p1, fpr* p2, uint8_t *reject);
__global__ void check_norm_u(int16_t* p1, uint32_t* p2, uint8_t *reject);
__global__ void poly_mul_fftx2(fpr *a, fpr *b, fpr *c, fpr *d);
__global__ void poly_mul_fft_add(fpr *a, fpr *b, fpr *c);
__global__ void reduce_mod(uint32_t *out, int16_t *in);
__global__ void poly_point_mul_ntt(uint32_t *out, uint16_t *in);
__global__ void NTT(uint32_t *A);
__global__ void iNTT(uint32_t *A);
__global__ void poly_add_u(uint32_t *a, uint32_t *b);
#endif
