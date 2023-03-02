#include <stdio.h>

#include "include/cuda_kernel.cuh"

// test vector for message m
// Now we use the same m, but we can copy a different m to each batch.
static uint8_t m_tv[MLEN] = {216, 28, 77, 141, 115, 79, 203, 251, 234, 222, 61, 63, 138, 3, 159, 170, 42, 44, 153, 87, 232, 53, 173, 85, 178, 46, 117, 191, 87, 187, 85, 106, 200};

int main() {
    uint8_t *h_sm, *h_m;
    uint32_t i, j;

    cudaMallocHost((void**) &h_sm, BATCH* (MLEN+CRYPTO_BYTES) * sizeof(uint8_t));
    cudaMallocHost((void**) &h_m, BATCH* (MLEN+CRYPTO_BYTES) * sizeof(uint8_t));

    for(j=0; j<BATCH; j++) for(i=0; i<MLEN; i++) h_m[j*MLEN + i] = m_tv[i];

    crypto_sign(h_sm, h_m);
    crypto_ver(h_sm, h_m);
    return 0;
}