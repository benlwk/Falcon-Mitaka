
// GPU settings
#define BATCH       1024
#define REPEAT      1
#define COMB_KER    // Combine multiple kernels for faster performance
#define NSAMP       512  // How many random samples per thread

// MITAKA parameters
#define MITAKA_D    512 // This code only support N=512
#define MITAKA_K    320
#define MITAKA_Q    12289
#define MSG_BYTES   32
#define R           1.32
#define R_SQUARE    1.7424

#if MITAKA_D == 512
  #define N MITAKA_D
  #define MITAKA_LOG_D 9
  #define GAMMA_SQUARE 100047795
  #define SIGMA_SQUARE 89985.416
// #elif MITAKA_D == 1024
//   #define N MITAKA_D
//   #define MITAKA_LOG_D 10
//   #define SIGMA_SQUARE 116245
//   #define GAMMA_SQUARE 258488745.55447942
#endif

// From Falcon
#define CRYPTO_BYTES            690
#define NONCELEN    40
#define MLEN        33

/* Constant-time macros */
#define LSBMASK(c)      (-((c)&1))
#define CMUX(x,y,c)     (((x)&(LSBMASK(c)))^((y)&(~LSBMASK(c))))
#define CFLIP(x,c)      CMUX(x,-(x),c)
#define CZERO64(x)      ((~(x)&((x)-1))>>63)

#include "../include/fpr.cuh"

void crypto_sign(fpr *h_c1, fpr *h_c2, uint8_t *h_mr);
void crypto_ver(fpr *h_pk, fpr *h_c1, fpr *h_c2, uint8_t *h_mr);





