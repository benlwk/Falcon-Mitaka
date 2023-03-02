
#define BATCH       1024
#define N           512
#define DEBUG
#define REPEAT      1
#define COMB_KER    // Combine multiple kernels for faster performance
#define LOGN        9
// #define DYN_PAR

/*
 * Constants for NTT.
 *
 *   n = 2^logn  (2 <= n <= 1024)
 *   phi = X^n + 1
 *   q = 12289
 *   q0i = -1/q mod 2^16
 *   R = 2^16 mod q
 *   R2 = 2^32 mod q
 */

#define Q           12289
#define Q0I         12287
#define R           4091
#define R2          10952

#define CRYPTO_SECRETKEYBYTES   1281
#define CRYPTO_PUBLICKEYBYTES   897
#define CRYPTO_BYTES            690

#define NONCELEN    40
#define MLEN        33  // fixed this for test vector verification.

#include <stdint.h>
void crypto_sign(uint8_t *h_sm, uint8_t *h_m);
void crypto_ver(uint8_t *h_sm, uint8_t *h_m);






