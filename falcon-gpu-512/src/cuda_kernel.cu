// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include <stdio.h>
#include "../include/cuda_kernel.cuh"
#include "../include/test_vector.cuh"
#include "../include/fpr.cuh"
#include "../include/fft.cuh"
#include "../include/shake.cuh"
#include "../include/ffSampling.cuh"
#include "../include/common.cuh"
#include "../include/rng.cuh"

// extern "C" 
void crypto_sign(uint8_t *h_sm, uint8_t *h_m) {
    cudaEvent_t start, stop;
    float elapsed = 0.0f;
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;

    int i, j;
    fpr *h_bb;
    fpr *d_bb;    
    // int8_t *h_F, *h_G, *h_f, *h_g;
    int8_t *d_F, *d_G, *d_f, *d_g;
    uint16_t *h_hm, *d_hm, *h_t1, *h_t2, *d_t1, *d_t2;
    int16_t *d_s1tmp, *d_s2tmp, *h_s1tmp, *h_s2tmp;
    uint64_t *h_scA, *d_scA, *h_scdptr, *d_scdptr;
    uint8_t *h_seed, *d_seed, *h_nonce, *d_nonce, *d_m, *h_esig, *d_esig, *d_sm, *d_sk, *h_sk;
    uint32_t *h_sqn, *d_sqn, *d_s, *h_s;
    uint32_t *d_seed_len, *h_seed_len, *d_nonce_len, *h_nonce_len, *h_esiglen, *d_esiglen;
    
    cudaMallocHost((void**) &h_hm, BATCH* N * sizeof(uint16_t));
    cudaMallocHost((void**) &h_s1tmp, BATCH* N * sizeof(int16_t));
    cudaMallocHost((void**) &h_s2tmp, BATCH* N * sizeof(int16_t));
    cudaMallocHost((void**) &h_bb, BATCH*10*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sqn, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_scA, BATCH*25*sizeof(uint64_t));
    cudaMallocHost((void**) &h_scdptr, BATCH*sizeof(uint64_t));
    cudaMallocHost((void**) &h_seed, BATCH*48*sizeof(uint8_t));
    cudaMallocHost((void**) &h_seed_len, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_nonce,BATCH*NONCELEN*sizeof(uint8_t));
    cudaMallocHost((void**) &h_nonce_len, BATCH*sizeof(uint32_t));
    
    cudaMallocHost((void**) &h_s, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_esig, BATCH*(CRYPTO_BYTES - 2 - NONCELEN)*sizeof(uint8_t));
    cudaMallocHost((void**) &h_esiglen, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_t1, BATCH*N*sizeof(uint16_t));
    cudaMallocHost((void**) &h_t2, BATCH*N*sizeof(uint16_t));
    cudaMallocHost((void**) &h_sk, BATCH* CRYPTO_SECRETKEYBYTES * sizeof(uint8_t));

    cudaMalloc((void**) &d_F, BATCH * N * sizeof(int8_t));
    cudaMalloc((void**) &d_G, BATCH * N * sizeof(int8_t));
    cudaMalloc((void**) &d_f, BATCH * N * sizeof(int8_t));
    cudaMalloc((void**) &d_g, BATCH * N * sizeof(int8_t));
    cudaMalloc((void**) &d_bb, BATCH*10*N*sizeof(fpr));
    cudaMalloc((void**) &d_hm, BATCH* N * sizeof(uint16_t));
    cudaMalloc((void**) &d_sqn, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_s1tmp, BATCH* N*sizeof(int16_t));
    cudaMalloc((void**) &d_s2tmp, BATCH* N*sizeof(int16_t));
    cudaMalloc((void**) &d_scA, BATCH*25*sizeof(uint64_t));
    cudaMalloc((void**) &d_scdptr, BATCH*sizeof(uint64_t));
    cudaMalloc((void**) &d_seed, BATCH*48*sizeof(uint8_t));
    cudaMalloc((void**) &d_seed_len,BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_nonce,BATCH*NONCELEN*sizeof(uint8_t));
    cudaMalloc((void**) &d_nonce_len, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_m, BATCH*MLEN*sizeof(uint8_t));
    cudaMalloc((void**) &d_s, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_esig, BATCH*(CRYPTO_BYTES - 2 - NONCELEN)*sizeof(uint8_t));
    cudaMalloc((void**) &d_esiglen, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_sm, BATCH* (MLEN+CRYPTO_BYTES) * sizeof(uint8_t));
    cudaMalloc((void**) &d_t1, BATCH*N*sizeof(uint16_t));
    cudaMalloc((void**) &d_t2, BATCH*N*sizeof(uint16_t));
    cudaMalloc((void**) &d_sk, BATCH* CRYPTO_SECRETKEYBYTES * sizeof(uint8_t));

    cudaMemset(d_bb, 0, BATCH*10*N*sizeof(fpr));
    cudaMemset(d_scdptr, 0, BATCH * sizeof(uint64_t));
    cudaMemset(d_hm, 0, BATCH*N*sizeof(uint16_t));

    for(j=0; j<BATCH; j++) 
    for(i=0; i<N; i++) 
    {
        h_hm[j*N + i] = hm[i];
        h_esig[j*(CRYPTO_BYTES-2-NONCELEN)+0] = 0x20 + 9;
    }

    // wklee, from randombytes(), which can generated on the CPU. Now we use the same seed for each batch, but in practical applications we can copy a different seed to each batch (j).   
    for(j=0; j<BATCH; j++) for(i=0; i<48; i++) h_seed[j*48 + i] = seed_tv[i];
    for(j=0; j<BATCH; j++) for(i=0; i<NONCELEN; i++) h_nonce[j*NONCELEN + i] = nonce_tv[i];
    for(j=0; j<BATCH; j++) for(i=0; i<CRYPTO_SECRETKEYBYTES; i++) h_sk[j*CRYPTO_SECRETKEYBYTES + i] = sk[i];

    cudaEventRecord(start);
    for(i=0; i<REPEAT; i++)    
    {
        cudaMemcpy(d_seed, h_seed, BATCH * 48* sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed_len, h_seed_len, BATCH *sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nonce, h_nonce, BATCH *NONCELEN*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, h_m, BATCH *MLEN*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_esig, h_esig, BATCH*(CRYPTO_BYTES - 2 - NONCELEN)*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk, h_sk, BATCH *CRYPTO_SECRETKEYBYTES*sizeof(uint8_t), cudaMemcpyHostToDevice);
        
        trim_i8_decode_gpu<<<BATCH, 1>>>(d_f, d_g, d_F, 9, max_fg_bits[9], d_sk + 1, CRYPTO_SECRETKEYBYTES - 1);

        // complete_private
#ifdef COMB_KER
        complete_private_comb_gpu<<<BATCH, N/2>>>(d_G, d_f, d_g, d_F, 9, d_t1, d_t2);
#else        
        complete_private_gpu<<<BATCH, N>>>(d_G, d_f, d_g, d_F, 9, d_t1, d_t2);
        mq_NTT_gpu<<<BATCH, N/2>>>(d_t1, LOGN);
        mq_NTT_gpu<<<BATCH, N/2>>>(d_t2, LOGN);
        mq_poly_tomonty<<<BATCH, N/2>>>(d_t1);
        mq_poly_montymul_ntt_gpu<<<BATCH, N>>>(d_t1, d_t2);
        mq_conv_small_gpu<<<BATCH, N>>>(d_t2, d_f);
        mq_NTT_gpu<<<BATCH, N/2>>>(d_t2, LOGN);
        complete_private_gpu2<<<BATCH, N>>>(d_t1, d_t2);
        mq_iNTT_gpu<<<BATCH, N/2>>>(d_t1);
        complete_private_gpu3<<<BATCH, N>>>(d_t1, d_G);

#endif        
        i_shake256_inject_gpu2<<<BATCH, 1>>>(d_scA, d_scdptr, d_nonce, NONCELEN);
        i_shake256_inject_gpu2<<<BATCH, 1>>>(d_scA, d_scdptr, d_m, MLEN);
        i_shake256_flip_gpu<<<BATCH, 1>>>(d_scA, d_scdptr);
        hash_to_point_vartime_par<<<BATCH,32>>>(d_scA, d_scdptr, d_hm);

        i_shake256_init_gpu<<<BATCH, 1>>>(d_scA, d_scdptr);
        i_shake256_inject_gpu2<<<BATCH, 1>>>(d_scA, d_scdptr, d_seed, 48);
        i_shake256_flip_gpu<<<BATCH, 1>>>(d_scA, d_scdptr);
                
        // b00 = tmp;    b01 = b00 + n;    b10 = b01 + n;
        // b11 = b10 + n;   t0 = b11 + n;    t1 = t0 + n;
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb, d_g, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+N, d_f, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+2*N, d_G, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+3*N, d_F, N, 10*N);
#ifdef COMB_KER
        FFT_SMx4_g<<<BATCH,128>>>(d_bb, d_bb+N, d_bb+2*N, d_bb+3*N, 10*N, 10*N);// b00 b01
#else        
        FFT_SM_g<<<BATCH,128>>>(d_bb, 10*N, 10*N);// b00
        FFT_SM_g<<<BATCH,128>>>(d_bb+N, 10*N, 10*N);// b01
        FFT_SM_g<<<BATCH,128>>>(d_bb+2*N, 10*N, 10*N);// b10
        FFT_SM_g<<<BATCH,128>>>(d_bb+3*N, 10*N, 10*N);// b11
#endif        
        
        poly_neg_g<<<BATCH,N>>>(d_bb+N, 10*N, 10*N);
        poly_neg_g<<<BATCH,N>>>(d_bb+3*N, 10*N, 10*N);

        poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+N);   // t0
        poly_copy<<<BATCH,N>>>(d_bb+5*N, d_bb);     // t1
        // //wklee, conserve b00 - b11, avoid recompute.
        // poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+N);   
        // poly_copy<<<BATCH,N>>>(d_bb+5*N, d_bb);    
        // poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+N);   
        // poly_copy<<<BATCH,N>>>(d_bb+5*N, d_bb);    
        poly_mulselfadj_fft_g<<<BATCH,N/2>>>(d_bb+4*N);
        poly_muladj_fft_g<<<BATCH,N/2>>>(d_bb+5*N, d_bb+2*N);
        poly_mulselfadj_fft_g<<<BATCH,N/2>>>(d_bb);
        poly_add_g<<<BATCH,N>>>(d_bb, d_bb+4*N);
        poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+N);   // t0
        poly_muladj_fft_g<<<BATCH,N/2>>>(d_bb+N, d_bb+3*N);
        poly_add_g<<<BATCH,N>>>(d_bb+N, d_bb+5*N);
        poly_mulselfadj_fft_g<<<BATCH,N/2>>>(d_bb+2*N);
        poly_copy<<<BATCH,N>>>(d_bb+5*N, d_bb+3*N);
        poly_mulselfadj_fft_g<<<BATCH,N/2>>>(d_bb+5*N);
        poly_add_g<<<BATCH,N>>>(d_bb+2*N, d_bb+5*N);
        
        //  * We rename variables to make things clearer. The three elements of the Gram matrix uses the first 3*n slots of tmp[], followed by b11 and b01 (in that order).
         
        // g00 = b00;    g01 = b01;    g11 = b10;
        // b01 = t0;    t0 = b01 + n;  t1 = t0 + n;    
        
        //  * Memory layout at that point:
        //  *   g00 g01 g11 b11 b01 t0 t1     
        //  * Set the target vector to [hm, 0] (hm is the hashed message).
         
        poly_set_g<<<BATCH,N>>>(d_bb+5*N, d_hm);    
        FFT_SM_g<<<BATCH,128>>>(d_bb+5*N, 10*N, 10*N);     
        poly_copy<<<BATCH,N>>>(d_bb+6*N, d_bb+5*N);
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+6*N, d_bb+4*N);
        poly_mulconst<<<BATCH,N>>>(d_bb+6*N, fpr_n(fpr_inverse_of_q));
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+5*N, d_bb+3*N);
        poly_mulconst<<<BATCH,N>>>(d_bb+5*N, fpr_inverse_of_q);

        /*
         * b01 and b11 can be discarded, so we move back (t0,t1).
         * Memory layout is now:
         *      g00 g01 g11 t0 t1
         */
        poly_copy<<<BATCH,N>>>(d_bb+3*N, d_bb+5*N); // t0=3*N
        poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+6*N); // t1=4*N
        

        ffSampling_fft_dyntree<<<BATCH,1>>>(d_bb+3*N, d_bb+4*N, d_bb, d_bb+1*N, d_bb+2*N, LOGN, LOGN, d_bb+5*N, d_scA, d_scdptr);        
    /*
     * We arrange the layout back to:
     *     b00 b01 b10 b11 t0 t1 tx ty    
     * We did not conserve the matrix basis, so we must recompute it now.
     */ 
        poly_copy<<<BATCH,N>>>(d_bb+5*N, d_bb+4*N); //t1   
        poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+3*N); // t0
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb, d_g, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+N, d_f, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+2*N, d_G, N, 10*N);
        smallints_to_fpr_g<<<BATCH, N>>>(d_bb+3*N, d_F, N, 10*N);
#ifdef COMB_KER        
        FFT_SMx4_g<<<BATCH,128>>>(d_bb, d_bb+N, d_bb+2*N, d_bb+3*N, 10*N, 10*N);
#else        
        FFT_SM_g<<<BATCH,128>>>(d_bb, 10*N, 10*N);   // b00
        FFT_SM_g<<<BATCH,128>>>(d_bb+N, 10*N, 10*N); // b01
        FFT_SM_g<<<BATCH,128>>>(d_bb+2*N, 10*N, 10*N);// b10
        FFT_SM_g<<<BATCH,128>>>(d_bb+3*N, 10*N, 10*N);// b11
#endif        
        poly_neg_g<<<BATCH,N>>>(d_bb+N, 10*N, 10*N);
        poly_neg_g<<<BATCH,N>>>(d_bb+3*N, 10*N, 10*N);
        poly_copy<<<BATCH,N>>>(d_bb+7*N, d_bb+5*N); //ty
        poly_copy<<<BATCH,N>>>(d_bb+6*N, d_bb+4*N); //tx
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+6*N, d_bb);
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+7*N, d_bb+2*N);
        poly_add_g<<<BATCH,N>>>(d_bb+6*N, d_bb+7*N);
        poly_copy<<<BATCH,N>>>(d_bb+7*N, d_bb+4*N); //t0->ty
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+7*N, d_bb+N);
        poly_copy<<<BATCH,N>>>(d_bb+4*N, d_bb+6*N); //tx->t0
        poly_mul_fft<<<BATCH,N/2>>>(d_bb+5*N, d_bb+3*N);
        poly_add_g<<<BATCH,N>>>(d_bb+5*N, d_bb+7*N);
        iFFT_g<<<BATCH,128>>>(d_bb+4*N, 10*N, 10*N);
        iFFT_g<<<BATCH,128>>>(d_bb+5*N, 10*N, 10*N);
        d_s1tmp = (int16_t*)d_bb+6*N;   // tx
        h_s1tmp = (int16_t*)h_bb+6*N;
        d_s2tmp = (int16_t*)d_bb;       // tmp          
        h_s2tmp = (int16_t*)h_bb;        
        check1<<<BATCH, 1>>>(d_s1tmp,d_bb+4*N, d_hm, d_sqn);
        check2<<<BATCH, N>>>(d_s2tmp, d_bb+5*N);
        // wklee, need to check if which block produces a short signature, reject it.
        is_short_half_gpu<<<BATCH,1>>>(d_sqn, d_s2tmp, d_s);
    /*
     * Encode the signature and bundle it with the message. Format is:
     *   signature length     2 bytes, big-endian
     *   nonce                40 bytes
     *   message              mlen bytes
     *   signature            slen bytes
     */
        comp_encode_gpu<<<BATCH, 1>>>(d_esig+1, (CRYPTO_BYTES - 2 - NONCELEN - 1), d_s2tmp, LOGN, d_esiglen);
        byte_copy<<<BATCH, MLEN>>>(d_sm + 2 + NONCELEN, d_m, (MLEN+CRYPTO_BYTES), MLEN);
        write_smlen_gpu<<<BATCH, 1>>>(d_sm, d_esiglen);
        byte_copy<<<BATCH, NONCELEN>>>(d_sm + 2, d_nonce, (MLEN+CRYPTO_BYTES), NONCELEN);
        byte_copy2<<<BATCH, 1>>>(d_sm + 2 + NONCELEN + MLEN, d_esig, (MLEN+CRYPTO_BYTES), d_esiglen);

        // cudaMemcpy(h_bb, d_bb, BATCH*7*N * sizeof(double), cudaMemcpyDeviceToHost);        
        // cudaMemcpy(h_scA, d_scA, BATCH*25*sizeof(uint64_t), cudaMemcpyDeviceToHost);  
        cudaMemcpy(h_sm, d_sm, BATCH*(MLEN+CRYPTO_BYTES)*sizeof(uint8_t), cudaMemcpyDeviceToHost);       
        // cudaMemcpy(h_esig, d_esig, BATCH*(CRYPTO_BYTES - 2 - NONCELEN)*sizeof(uint8_t), cudaMemcpyDeviceToHost);    
        // cudaMemcpy(h_esiglen, d_esiglen, BATCH*sizeof(uint32_t), cudaMemcpyDeviceToHost);     
        // cudaMemcpy(h_s2tmp, d_s2tmp, BATCH*N*sizeof(int16_t), cudaMemcpyDeviceToHost);                
        // cudaMemcpy(h_f, d_f, BATCH*N* sizeof(int8_t), cudaMemcpyDeviceToHost);  
        // cudaMemcpy(h_t1, d_t1, BATCH*N* sizeof(uint16_t), cudaMemcpyDeviceToHost);      
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ; 
    printf("\nTotal time: %.4f ms, TP: %.0f \n", elapsed/REPEAT, 1000*BATCH/(elapsed/REPEAT));

    
#ifdef DEBUG
    for(j=0; j<BATCH; j++) for(i=0; i<(MLEN+CRYPTO_BYTES); i++)  {
        if(h_sm[j*(MLEN+CRYPTO_BYTES) + i]!= h_sm[i]){
            printf("Wrong at batch %u loc %u: %u %u\n", j, i, h_sm[j*(MLEN+CRYPTO_BYTES) + i], h_sm[i]);
            break;
        }        
    }    
#endif
}



// extern "C" 
void crypto_ver(uint8_t *h_sm, uint8_t *h_m) {
    cudaEvent_t start, stop;
    float elapsed = 0.0f;
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;

    int i, j;

    uint16_t *h_tmp, *d_tmp;
    int16_t *h_c0, *h_h;
    uint16_t *d_c0, *d_h, *d_hm;
    int16_t *h_s2, *d_s2, *h_sig, *d_sig;
    uint32_t *d_s, *h_s, *h_smlen, *d_smlen, *h_msg_len, *d_msg_len, *d_sig_len;
    uint8_t *h_pk, *d_pk, *d_sm, *d_m;    
    // inner_shake256_context *h_sc, *d_sc;
    uint64_t *h_scA, *d_scA, *h_scdptr, *d_scdptr, *h_test, *d_test, *d_test2;

    cudaMallocHost((void**) &h_c0, BATCH* N * sizeof(uint16_t));
    cudaMallocHost((void**) &h_tmp, BATCH* N * sizeof(uint16_t));
    cudaMallocHost((void**) &h_s2, BATCH* N * sizeof(int16_t));
    cudaMallocHost((void**) &h_h, BATCH* N * sizeof(uint16_t));
    cudaMallocHost((void**) &h_s, BATCH* sizeof(uint32_t));
    cudaMallocHost((void**) &h_pk, BATCH*CRYPTO_PUBLICKEYBYTES* sizeof(uint8_t));
    // cudaMallocHost((void**) &h_sm, BATCH*(MLEN+CRYPTO_BYTES)* sizeof(uint8_t));
    cudaMallocHost((void**) &h_smlen, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_msg_len, BATCH*sizeof(uint32_t));
    cudaMallocHost((void**) &h_sig, BATCH* N * sizeof(int16_t));
    cudaMallocHost((void**) &h_scA, BATCH* 25 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_test, BATCH* 25 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_scdptr, BATCH * sizeof(uint64_t));

    cudaMalloc((void**) &d_c0, BATCH* N * sizeof(uint16_t));
    cudaMalloc((void**) &d_tmp, BATCH*N*sizeof(uint16_t));
    cudaMalloc((void**) &d_s2, BATCH* N * sizeof(int16_t));
    cudaMalloc((void**) &d_h, BATCH* N * sizeof(uint16_t));
    cudaMalloc((void**) &d_s, BATCH* sizeof(uint32_t));
    cudaMalloc((void**) &d_pk, BATCH*CRYPTO_PUBLICKEYBYTES* sizeof(uint8_t));
    cudaMalloc((void**) &d_sm, BATCH*(MLEN+CRYPTO_BYTES)* sizeof(uint8_t));
    cudaMalloc((void**) &d_smlen, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_msg_len, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_sig_len, BATCH*sizeof(uint32_t));
    cudaMalloc((void**) &d_sig, BATCH* N * sizeof(int16_t));
    cudaMalloc((void**) &d_scA, BATCH* 25 * sizeof(uint64_t));
    cudaMalloc((void**) &d_scdptr, BATCH * sizeof(uint64_t));
    cudaMalloc((void**) &d_hm, BATCH* N * sizeof(uint16_t));
    cudaMalloc((void**) &d_test, BATCH* 25 * sizeof(uint64_t));
    cudaMalloc((void**) &d_test2, BATCH* 25 * sizeof(uint64_t));
    cudaMalloc((void**) &d_m, BATCH* 25 * sizeof(uint64_t));

    cudaMemset(d_scdptr, 0, BATCH * sizeof(uint64_t));
    cudaMemset(d_hm, 0, BATCH * sizeof(uint16_t));
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaFuncSetCacheConfig(hash_to_point_vartime_par, cudaFuncCachePreferShared);   
 

    for(j=0; j<BATCH; j++) 
    for(i=0; i<N; i++) 
    {
        h_h[j*N + i] = h[i];    
    }

    for(j=0; j<BATCH; j++) 
        for(i=0; i<CRYPTO_PUBLICKEYBYTES; i++){             
            h_pk[j*CRYPTO_PUBLICKEYBYTES + i] = pk[i];    
        }

    for(j=0; j<BATCH; j++) {
        h_smlen[j] = 691; // wklee, change to variable
    }

    for(j=0; j<25; j++) h_test[j] = 2*j;

    cudaEventRecord(start);
    for(i=0; i<REPEAT; i++)    
    {        
        // cudaMemcpy(d_c0, h_c0, BATCH * N * sizeof(uint16_t), cudaMemcpyHostToDevice);    
        // cudaMemcpy(d_s2, h_s2, BATCH * N * sizeof(int16_t), cudaMemcpyHostToDevice);    
        // cudaMemcpy(d_h, h_h, BATCH * N * sizeof(uint16_t), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_pk, h_pk, BATCH * CRYPTO_PUBLICKEYBYTES * sizeof(uint8_t), cudaMemcpyHostToDevice);   
        cudaMemcpy(d_sm, h_sm, BATCH * (MLEN + CRYPTO_BYTES) * sizeof(uint8_t), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_test, h_test, BATCH * 25* sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_smlen, h_smlen, BATCH * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_msg_len, h_msg_len, BATCH * sizeof(uint32_t), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_m, h_m, BATCH * MLEN * sizeof(uint8_t), cudaMemcpyHostToDevice);        

        modq_decode_gpu<<<BATCH, N/4>>>(d_h, LOGN, d_pk+1, CRYPTO_PUBLICKEYBYTES - 1);
        mq_NTT_gpu<<<BATCH, N/2>>>(d_h, LOGN);
        mq_poly_tomonty<<<BATCH, N/2>>>(d_h);
        msg_len_gpu<<<BATCH, 1>>>(d_sm, d_msg_len, d_smlen, d_sig_len);
    /*
     * Decode signature.
     */
        // comp_decode(sig, 9, esig + 1, sig_len - 1) 
        comp_decode_gpu<<<BATCH, 1>>>(d_sig, 9, d_sm, d_sig_len, d_msg_len);
        // i_shake256_init_gpu<<<BATCH, 1>>>(d_sc);
        i_shake256_inject_gpu<<<BATCH, 1>>>(d_scA, d_scdptr, d_sm + 2, d_msg_len);
        i_shake256_flip_gpu<<<BATCH, 1>>>(d_scA, d_scdptr);
        hash_to_point_vartime_par<<<BATCH,32>>>(d_scA, d_scdptr, d_hm);
        // below are from verify_raw()
#ifdef COMB_KER        
        comb_all_kernels<<<BATCH, N/2>>>(d_tmp, d_sig, d_h, d_hm);
#else
        reduce_s2<<<BATCH, N>>>(d_sig, d_tmp); 
        mq_NTT_gpu<<<BATCH, N/2>>>(d_tmp, LOGN);
        mq_poly_montymul_ntt_gpu<<<BATCH, N>>>(d_tmp, d_h);
        mq_iNTT_gpu<<<BATCH, N/2>>>(d_tmp);
        mq_poly_sub<<<BATCH, N>>>(d_tmp, d_hm);
        norm_s2<<<BATCH, N>>>(d_tmp);   
#endif  

        // wklee, need to check which block produces a short signature, reject it.
        is_short_gpu<<<BATCH, 1>>>((int16_t *)d_tmp, d_sig, d_s);
        /*
        * Return plaintext.
        */
        // memmove(m, sm + 2 + NONCELEN, msg_len);
        byte_cmp<<<BATCH, MLEN>>>(d_m, d_sm + 2 + NONCELEN);

        // cudaMemcpy(h_sm, d_sm, BATCH*(MLEN+CRYPTO_BYTES) * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
        // cudaMemcpy(h_scA, d_scA, BATCH*25 * sizeof(uint64_t), cudaMemcpyDeviceToHost);      
        // cudaMemcpy(h_s, d_s, BATCH*sizeof(uint64_t), cudaMemcpyDeviceToHost); 
        // cudaMemcpy(h_h, d_h, BATCH* N*sizeof(int16_t), cudaMemcpyDeviceToHost);          
        cudaMemcpy(h_tmp, d_tmp, BATCH* N*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_test, d_scA, BATCH* 25*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ;
    // totaltime += elapsed;    
    printf("\nTotal time: %.4f ms, TP: %.0f \n", elapsed/REPEAT, 1000*BATCH/(elapsed/REPEAT));

    
#ifdef DEBUG
    for(j=0; j<BATCH; j++) for(i=0; i<N; i++)  {
        if(h_tmp[j*N + i]!= h_tmp[i]){
            printf("Wrong at batch %u loc %u: %u %u\n", j, i, h_tmp[j*N + i], h_tmp[i]);
            break;
        }        
    }    
#endif
}











