// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
#include "../include/test_vector.cuh"
#include "../include/fft.cuh"
#include <stdio.h>
#include "../include/shake.cuh"
#include "../include/samplerZ.cuh"

void crypto_sign(fpr *h_c1, fpr *h_c2, uint8_t *h_mr) {
    cudaEvent_t start, stop;
    float elapsed = 0.0f, totaltime = 0.0f;
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;

    int i, j;
    fpr *d_c1, *d_c2, *h_vec_y1, *d_vec_y1, *h_vec_y2, *d_vec_y2, *h_d, *d_d, *h_temp, *d_temp, *d_x, *h_gauss_x, *d_gauss_x, *d_v0, *d_v1, *h_gauss_x2, *d_gauss_x2;    
    uint64_t *h_u_y1, *h_v_y1, *h_e_y1, *h_u_y2, *h_v_y2, *h_e_y2;
    uint64_t *d_u_y1, *d_v_y1, *d_e_y1, *d_u_y2, *d_v_y2, *d_e_y2;
    uint8_t *d_mr, *h_prng, *d_prng, *h_rej, *d_rej, *d_prng_key, *h_prng_key;
    uint64_t *h_tmp, *d_tmp;
    fpr *h_sigma1, *h_sigma2, *h_sk_b10, *h_sk_b11, *h_sk_b20, *h_sk_b21, *h_beta10, *h_beta11, *h_beta20, *h_beta21, *h_tempc1, *h_tempc2;
    fpr *d_sigma1, *d_sigma2, *d_sk_b10, *d_sk_b11, *d_sk_b20, *d_sk_b21, *d_beta10, *d_beta11, *d_beta20, *d_beta21, *d_tempc1, *d_tempc2;

    cudaMallocHost((void**) &h_tmp, BATCH*25* sizeof(uint64_t));
    cudaMallocHost((void**) &h_prng, (uint64_t) BATCH*N*NSAMP*sizeof(uint8_t));
    cudaMallocHost((void**) &h_prng_key, BATCH*56*sizeof(uint8_t));
    cudaMallocHost((void**) &h_u_y1, BATCH* N/2 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_v_y1, BATCH* N/2 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_e_y1, BATCH* N * sizeof(uint64_t));
    cudaMallocHost((void**) &h_vec_y1, BATCH* N * sizeof(fpr));
    cudaMallocHost((void**) &h_u_y2, BATCH* N/2 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_v_y2, BATCH* N/2 * sizeof(uint64_t));
    cudaMallocHost((void**) &h_e_y2, BATCH* N * sizeof(uint64_t));
    cudaMallocHost((void**) &h_vec_y2, BATCH* N * sizeof(fpr));
    cudaMallocHost((void**) &h_sigma1, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sigma2, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sk_b10, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sk_b11, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sk_b20, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_sk_b21, BATCH*N*sizeof(fpr));
    
    cudaMallocHost((void**) &h_beta10, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_beta11, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_beta20, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_beta21, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_tempc1, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_tempc2, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_d, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_temp, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_gauss_x, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_gauss_x2, BATCH*N*sizeof(fpr));
    cudaMallocHost((void**) &h_rej, BATCH*sizeof(uint8_t));

    cudaMalloc((void**) &d_mr, BATCH* (MSG_BYTES+MITAKA_K/8) * sizeof(uint8_t));
    cudaMalloc((void**) &d_tmp, BATCH*25* sizeof(uint64_t));
    cudaMalloc((void**) &d_c1, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_c2, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_prng, (uint64_t) BATCH*N*NSAMP*sizeof(uint8_t));
    cudaMalloc((void**) &d_prng_key, BATCH*56*sizeof(uint8_t));
    
    cudaMalloc((void**) &d_u_y1, BATCH* N/2 * sizeof(uint64_t));
    cudaMalloc((void**) &d_v_y1, BATCH* N/2 * sizeof(uint64_t));
    cudaMalloc((void**) &d_e_y1, BATCH* N * sizeof(uint64_t));
    cudaMalloc((void**) &d_vec_y1, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_u_y2, BATCH* N/2 * sizeof(uint64_t));
    cudaMalloc((void**) &d_v_y2, BATCH* N/2 * sizeof(uint64_t));
    cudaMalloc((void**) &d_e_y2, BATCH* N * sizeof(uint64_t));
    cudaMalloc((void**) &d_vec_y2, BATCH* N * sizeof(fpr)); 
    cudaMalloc((void**) &d_sigma1, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_sigma2, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_sk_b10, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_sk_b11, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_sk_b20, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_sk_b21, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_beta10, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_beta11, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_beta20, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_beta21, BATCH*N*sizeof(fpr));   
    cudaMalloc((void**) &d_tempc1, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_tempc2, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_d, BATCH*N*sizeof(fpr));        
    cudaMalloc((void**) &d_x, BATCH*N*sizeof(fpr));   
    cudaMalloc((void**) &d_temp, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_gauss_x, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_v0, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_v1, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_gauss_x2, BATCH*N*sizeof(fpr));
    cudaMalloc((void**) &d_rej, BATCH*sizeof(uint8_t));

    // cudaMemset(d_bb, 0, BATCH*10*N*sizeof(fpr));

    for(j=0; j<BATCH; j++) 
    for(i=0; i<N/2; i++) 
    {
        h_u_y1[j*N/2 + i] = u_y1[i];
        h_v_y1[j*N/2 + i] = v_y1[i];
        h_e_y1[j*N + i] = e_y1[i];        
        h_e_y1[j*N + i+N/2] = e_y1[i+N/2];  
        h_u_y2[j*N/2 + i] = u_y2[i];
        h_v_y2[j*N/2 + i] = v_y2[i];
        h_e_y2[j*N + i] = e_y2[i];        
        h_e_y2[j*N + i+N/2] = e_y2[i+N/2];          
    }

    for(j=0; j<BATCH; j++) 
    for(i=0; i<MSG_BYTES+MITAKA_K/8; i++) 
    {
        h_mr[j*(MSG_BYTES+MITAKA_K/8) + i] = m[i];
    }
    
    // wklee, Secret Key
    // Right now we use the same key, but it is easy to use a different key. Just copy them on to the respective batch (j).
    for(j=0; j<BATCH; j++) 
    for(i=0; i<N; i++) 
    {
        h_sigma1[j*N + i].v = sigma1[i];          
        h_sigma2[j*N + i].v = sigma2[i];   
        h_sk_b10[j*N + i].v = sk_b10[i];          
        h_sk_b11[j*N + i].v = sk_b11[i];   
        h_sk_b20[j*N + i].v = sk_b20[i];          
        h_sk_b21[j*N + i].v = sk_b21[i];   
        h_beta10[j*N + i].v = beta10[i];          
        h_beta11[j*N + i].v = beta11[i];   
        h_beta20[j*N + i].v = beta20[i];          
        h_beta21[j*N + i].v = beta21[i];           
        h_gauss_x[j*N + i].v = gauss_x[i];
        h_gauss_x2[j*N + i].v = gauss_x2[i];
    }
    
    for(j=0; j<BATCH; j++) 
        for(i=0; i<56; i++) 
            h_prng_key[j*56 + i] = rand() & 0xFF;

    
    for(i=0; i<REPEAT; i++)    
    {
        cudaEventRecord(start);
        cudaMemcpy(d_mr, h_mr, BATCH*(MSG_BYTES+MITAKA_K/8)*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_y1, h_u_y1, BATCH*N/2*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v_y1, h_v_y1, BATCH*N/2*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_e_y1, h_e_y1, BATCH*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_y2, h_u_y2, BATCH*N/2*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v_y2, h_v_y2, BATCH*N/2*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_e_y2, h_e_y2, BATCH*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sigma1, h_sigma1, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sigma2, h_sigma2, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk_b10, h_sk_b10, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk_b11, h_sk_b11, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk_b20, h_sk_b20, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk_b21, h_sk_b21, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);        
        cudaMemcpy(d_beta10, h_beta10, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta11, h_beta11, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta20, h_beta20, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta21, h_beta21, BATCH*N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prng_key, h_prng_key, BATCH*56*sizeof(uint8_t), cudaMemcpyHostToDevice);   

        shake128_absorb_gpu<<<BATCH, 25>>>(d_tmp, d_mr, MSG_BYTES+MITAKA_K/8);
        shake128_squeezeblocks<<<BATCH, 25>>>(d_c2, d_tmp);
        prng_refill_g<<<BATCH, N>>>(d_prng, d_prng_key);
#ifdef COMB_KER
        normaldist_mul_fft_g<<<BATCH, N/2>>>(d_vec_y1, d_u_y1, d_v_y1, d_e_y1, d_sigma1);
        normaldist_mul_fft_g<<<BATCH, N/2>>>(d_vec_y2, d_u_y2, d_v_y2, d_e_y2, d_sigma2);
#else        
        normaldist_g<<<BATCH, N/2>>>(d_vec_y1, d_u_y1, d_v_y1, d_e_y1);
        normaldist_g<<<BATCH, N/2>>>(d_vec_y2, d_u_y2, d_v_y2, d_e_y2);
        poly_mul_fft<<<BATCH,N/2>>>(d_vec_y1, d_sigma1);
        poly_mul_fft<<<BATCH,N/2>>>(d_vec_y2, d_sigma2);
#endif        
        // poly_copy<<<BATCH,N>>>(d_tempc1, d_c1);
        // FFT_SM_g<<<BATCH,N/2>>>(d_tempc1);// skip, zero
        poly_copy<<<BATCH,N>>>(d_tempc2, d_c2);
        FFT_SM_g<<<BATCH,N/4>>>(d_tempc2, N, N);
        // poly_copy<<<BATCH,N>>>(d_d, d_c1);// skip, zero
        // poly_mul_fft<<<BATCH,N/2>>>(d_s, d_beta20);
        poly_copy<<<BATCH,N>>>(d_temp, d_tempc2);
#ifdef COMB_KER
        poly_mul_fft_add<<<BATCH,N/2>>>(d_temp, d_beta21, d_d);
#else        
        poly_mul_fft<<<BATCH,N/2>>>(d_temp, d_beta21);
        poly_add_g<<<BATCH,N>>>(d_d, d_temp);
#endif
        poly_copy<<<BATCH,N>>>(d_x, d_d);
        poly_sub_g<<<BATCH,N>>>(d_x, d_vec_y2);
        iFFT_g<<<BATCH,N/4>>>(d_x, N, N);
        sample_discrete_gauss_gpu<<<BATCH,N>>>(d_gauss_x, d_prng);
        FFT_SM_g<<<BATCH,N/4>>>(d_gauss_x, N, N);
        poly_copy<<<BATCH,N>>>(d_v0, d_gauss_x);
        poly_copy<<<BATCH,N>>>(d_v1, d_gauss_x);
#ifdef COMB_KER
        poly_mul_fftx2<<<BATCH,N/2>>>(d_v0, d_sk_b20, d_v1, d_sk_b21);
#else        
        poly_mul_fft<<<BATCH,N/2>>>(d_v0, d_sk_b20);
        poly_mul_fft<<<BATCH,N/2>>>(d_v1, d_sk_b21);
#endif        
        poly_sub_g<<<BATCH,N>>>(d_tempc1, d_v0);
        poly_sub_g<<<BATCH,N>>>(d_tempc2, d_v1);
        poly_copy<<<BATCH,N>>>(d_d, d_tempc1);
        poly_copy<<<BATCH,N>>>(d_temp, d_tempc2);  
#ifdef COMB_KER   
        poly_mul_fftx2<<<BATCH,N/2>>>(d_d, d_beta10, d_temp, d_beta11);
#else       
        poly_mul_fft<<<BATCH,N/2>>>(d_d, d_beta10);      
        poly_mul_fft<<<BATCH,N/2>>>(d_temp, d_beta11);      
#endif             
        poly_add_g<<<BATCH,N>>>(d_d, d_temp);
        poly_copy<<<BATCH,N>>>(d_x, d_d);
        poly_sub_g<<<BATCH,N>>>(d_x, d_vec_y1);
        iFFT_g<<<BATCH,N/4>>>(d_x, N, N);
        // Use the second half of the random samples
        sample_discrete_gauss_gpu<<<BATCH,N>>>(d_gauss_x2, d_prng+NSAMP/2); 
        FFT_SM_g<<<BATCH,N/4>>>(d_gauss_x2, N, N);
        poly_copy<<<BATCH,N>>>(d_temp, d_gauss_x2);
#ifdef COMB_KER
        poly_mul_fft_add<<<BATCH,N/2>>>(d_temp, d_sk_b10, d_v0);
        poly_mul_fft_add<<<BATCH,N/2>>>(d_gauss_x2, d_sk_b11, d_v1);
#else                
        poly_mul_fft<<<BATCH,N/2>>>(d_temp, d_sk_b10);
        poly_add_g<<<BATCH,N>>>(d_v0, d_temp);
        poly_mul_fft<<<BATCH,N/2>>>(d_gauss_x2, d_sk_b11);
        poly_add_g<<<BATCH,N>>>(d_v1, d_gauss_x2);
#endif
        iFFT_g<<<BATCH,N/4>>>(d_v0, N, N);
        iFFT_g<<<BATCH,N/4>>>(d_v1, N, N);
        poly_sub_g<<<BATCH,N>>>(d_c1, d_v0);
        poly_sub_g<<<BATCH,N>>>(d_c2, d_v1);
        poly_recenter<<<BATCH,N>>>(d_c1);
        poly_recenter<<<BATCH,N>>>(d_c2);
        check_norm<<<BATCH,1>>>(d_c1, d_c2, d_rej);
        
        cudaMemcpy(h_c1, d_c1, BATCH*N*sizeof(fpr), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c2, d_c2, BATCH*N*sizeof(fpr), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop) ;
        totaltime += elapsed;   
        cudaMemset(d_prng, 0, (uint64_t)BATCH*N*NSAMP*sizeof(uint8_t));     
        cudaMemset(d_gauss_x, 0, BATCH*N*sizeof(fpr));     
        cudaMemset(d_gauss_x2, 0, BATCH*N*sizeof(fpr));             
    }
    printf("\nTotal time: %.4f ms, TP: %.0f \n", totaltime/REPEAT, 1000*BATCH/(totaltime/REPEAT));

    
#ifdef DEBUG
    for(j=0; j<BATCH; j++) for(i=0; i<N; i++)  {
        if(h_c1[j*N + i].v!= h_c1[i].v){
            printf("Wrong at batch %u loc %u: %.4f %.4f\n", j, i, h_c1[j*N + i].v, h_c1[i].v);
            break;
        }        
    }  
#endif
}


// h_mr contains message m + signature->r
// h_c1 contains s1, h_c2 contains s2
void crypto_ver(fpr *h_pk, fpr *h_c1, fpr *h_c2, uint8_t *h_mr) {
    cudaEvent_t start, stop;
    float elapsed = 0.0f, totaltime = 0.0f;
    cudaEventCreate(&start);    cudaEventCreate(&stop) ;

    int i, j;
    uint8_t *d_mr, *h_rej, *d_rej;
    fpr *d_pk, *h_cc1, *d_cc1, *h_t, *d_t, *d_c1;
    uint64_t *d_tmp;

    cudaMallocHost((void**) &h_cc1, BATCH* N * sizeof(fpr));
    cudaMallocHost((void**) &h_t, BATCH* N * sizeof(fpr));
    cudaMallocHost((void**) &h_rej, BATCH*sizeof(uint8_t));

    cudaMalloc((void**) &d_t, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_pk, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_cc1, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_c1, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_pk, BATCH* N * sizeof(fpr));
    cudaMalloc((void**) &d_tmp, BATCH*25* sizeof(uint64_t));
    cudaMalloc((void**) &d_mr, BATCH* (MSG_BYTES+MITAKA_K/8) * sizeof(uint8_t));
    cudaMalloc((void**) &d_rej, BATCH*sizeof(uint8_t));
    cudaMemset(d_tmp, 0, BATCH*25* sizeof(uint64_t));

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    // cudaFuncSetCacheConfig(hash_to_point_vartime_par, cudaFuncCachePreferShared);   

    for(j=0; j<BATCH; j++) 
    for(i=0; i<N; i++) 
    {
        h_pk[j*N + i].v = pk[i];
    }

    cudaEventRecord(start); 
    for(i=0; i<REPEAT; i++)    
    {        
        cudaMemcpy(d_mr, h_mr, BATCH*(MSG_BYTES+MITAKA_K/8)*sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c1, h_c1, BATCH* N*sizeof(fpr), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pk, h_pk, BATCH* N*sizeof(fpr), cudaMemcpyHostToDevice);

        shake128_absorb_gpu2<<<BATCH, 25>>>(d_tmp, d_mr, MSG_BYTES+MITAKA_K/8);
        shake128_squeezeblocks<<<BATCH, 25>>>(d_cc1, d_tmp);
        poly_copy<<<BATCH,N>>>(d_t, d_c1);
        FFT_SM_g<<<BATCH,N/4>>>(d_t, N, N);
        poly_mul_fft<<<BATCH,N/2>>>(d_t, d_pk);
        iFFT_g<<<BATCH,N/4>>>(d_t, N, N);
        poly_add_g<<<BATCH,N>>>(d_t, d_cc1);
        poly_recenter<<<BATCH,N>>>(d_t);
        check_norm<<<BATCH,1>>>(d_c1, d_t, d_rej);

        cudaMemcpy(h_t, d_t, BATCH* N*sizeof(fpr), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop) ;
    printf("\nTotal time: %.4f ms, TP: %.0f \n", elapsed/REPEAT, 1000*BATCH/(elapsed/REPEAT));
   
#ifdef DEBUG
    for(j=0; j<BATCH; j++) for(i=0; i<N; i++)  {
        if(h_t[j*N + i].v!= h_t[i].v){
            printf("Wrong at batch %u loc %u: %f %f\n", j, i, h_t[j*N + i].v, h_t[i].v);
            break;
        }        
    } 
#endif
}











