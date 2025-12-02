#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "tvm/runtime/registry.h"

#include "kernel_params_list.hpp"

#include "autogemm_sgemm.h"

void* _mm_malloc(size_t align, size_t sz)
{
  void *ptr;
  int alloc_result = posix_memalign(&ptr, align, sz);
  if(alloc_result != 0)
  {
    return NULL;
  }
  return ptr;
}

void autogemm_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
    const BLASINT m, const BLASINT n, const BLASINT k, const float alpha, const float *a, const BLASINT lda,
    const float *b, const BLASINT ldb, const float beta, float *c, const BLASINT ldc)
{
    // void(transA); // only support N
    // void(transB); // only support N

    if (lda != k) {
        printf("ldA != k\n");
        return;
    }
    if (ldb != n) {
        printf("ldB ! = n\n");
        return;
    }
    if (ldc != n) {
        printf("ldC ! = n\n");
        return;
    }

    int M = m;
    int N = n;
    int K = k;

    int packedB_size;

    KernelParams::CreateMap();
    KernelParams::Key query_key = {M, N, K};
    auto it = KernelParams::mapping.find(query_key)->second;
    packedB_size = it.packedB_size;


    // printf("MxNxK = %dx%dx%d\n", m, n, k);
    // printf("m = %d, n = %d, k = %d, nc = %d, kc = %d, padding_size = %d， nc_ceil = %d\n", m, n, k, nc, kc, padding_size, nc_ceil);
    // printf("Allocating packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));

    float *packedB = static_cast<float*>(_mm_malloc(64, packedB_size * sizeof(float)));

    tvm::runtime::PackedFunc pack_func = it.pack_func;
    tvm::runtime::PackedFunc func = it.func;

    // printf("Begin allocating DLTensors\n");
    DLTensor tvm_A;
    DLTensor tvm_B;
    DLTensor tvm_packedB;
    DLTensor tvm_C;

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    DLDataType dtype = {kDLFloat, dtype_bits, dtype_lanes};
    DLDevice device = {kDLCPU, device_id};

    // printf("Setting device\n");
    tvm_A.device = device;
    tvm_B.device = device;
    tvm_packedB.device = device;
    tvm_C.device = device;

    // printf("Setting dtype\n");
    tvm_A.dtype = dtype;
    tvm_B.dtype = dtype;
    tvm_packedB.dtype = dtype;
    tvm_C.dtype = dtype;

    // printf("Setting ndim\n");
    tvm_A.ndim = 2;
    tvm_B.ndim = 2;
    tvm_packedB.ndim = 4;
    tvm_C.ndim = 2;

    // printf("Setting shape\n");
    tvm_A.shape = it.A_shape;
    tvm_B.shape = it.B_shape;
    tvm_packedB.shape = it.packedB_shape;
    tvm_C.shape = it.C_shape;

    // printf("Setting data\n");
    tvm_A.data = (void *)a;
    tvm_B.data = (void *)b;
    tvm_packedB.data = packedB;
    tvm_C.data = (void *)c;

    tvm_A.strides = nullptr;
    tvm_B.strides = nullptr;
    tvm_packedB.strides = nullptr;
    tvm_C.strides = nullptr;

    tvm_A.byte_offset = 0;
    tvm_B.byte_offset = 0;
    tvm_packedB.byte_offset = 0;
    tvm_C.byte_offset = 0;

    // printf("check a\n");
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < k; j++) {
    //     ((float *)a)[i * k + j] = 0.0;
    //   }
    // }

    // printf("check c\n");
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < n; j++) {
    //     ((float *)c)[i * n + j] = 0.0;
    //   }
    // }

    // printf("packed_func and func executing\n");
    pack_func(&tvm_B, &tvm_packedB);
    // printf("packed_func execution done\n");
    func(&tvm_A, &tvm_packedB, &tvm_C);
    // printf("packed_func and func execution done\n");

    // pack_func(b, packedB);
    // func(a, packedB, c);

    free(packedB);
    // printf("Free packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));
}

#endif