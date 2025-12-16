#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <chrono>

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

    // printf("MxNxK = %dx%dx%d\n", m, n, k);

    KernelParams::CreateMap();
    std::string query_key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    auto it = KernelParams::mapping.find(query_key)->second;

    tvm::runtime::PackedFunc func = it.func;

    DLTensor tvm_A;
    DLTensor tvm_B;
    DLTensor tvm_C;

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    DLDataType dtype = {kDLFloat, dtype_bits, dtype_lanes};
    DLDevice device = {kDLCPU, device_id};

    tvm_A.device = device;
    tvm_B.device = device;
    tvm_C.device = device;

    // printf("Setting dtype\n");
    tvm_A.dtype = dtype;
    tvm_B.dtype = dtype;
    tvm_C.dtype = dtype;

    // printf("Setting ndim\n");
    tvm_A.ndim = 2;
    tvm_B.ndim = 2;
    tvm_C.ndim = 2;

    // printf("Setting shape\n");
    tvm_A.shape = it.A_shape;
    tvm_B.shape = it.B_shape;
    tvm_C.shape = it.C_shape;

    // printf("Setting data\n");
    tvm_A.data = (void *)a;
    tvm_B.data = (void *)b;
    tvm_C.data = (void *)c;

    tvm_A.strides = nullptr;
    tvm_B.strides = nullptr;
    tvm_C.strides = nullptr;

    tvm_A.byte_offset = 0;
    tvm_B.byte_offset = 0;
    tvm_C.byte_offset = 0;

    func(&tvm_A, &tvm_B, &tvm_C);
}

#endif