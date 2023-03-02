# Falcon-Mitaka
GPU Implementation of Falcon and Mitaka Signature Schemes

This is the repository for the codes accompanying the paper "High Throughput Lattice-based Signatures on
GPUs: Comparing Falcon and Mitaka". You can find the paper here:

The released source codes only support Falcon-512 and Mitaka-512. You can compile the codes through following steps:

1) Look at Makefile,
- Change the artitecture that matches your GPU (-arch sm_86). Currently it supports CUDA compute capability 8.6, which is tested on RTX 3080 and A100.
- Change the CUDA path (CUDA_ROOT_DIR) and NVCC.

2) Type "make" to compile and "clean" to remove all binaries.
