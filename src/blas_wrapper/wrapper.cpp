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

    tvm::runtime::PackedFunc pack_func = it.pack_func;
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

    int packAB = it.packAB;
    if (packAB == 0) {
        func(&tvm_A, &tvm_B, &tvm_C);
    } else if (packAB == 1) {
        int packedA_size = it.packedA_size;
        float *packedA = static_cast<float*>(_mm_malloc(64, packedA_size * sizeof(float)));

        // printf("Begin allocating DLTensors\n");
        DLTensor tvm_packedA;
        tvm_packedA.device = device;
        tvm_packedA.dtype = dtype;
        tvm_packedA.ndim = 4;
        tvm_packedA.shape = it.packedA_shape;
        tvm_packedA.data = packedA;
        tvm_packedA.strides = nullptr;
        tvm_packedA.byte_offset = 0;
        // printf("packed_func and func executing\n");
        // auto start_time = std::chrono::steady_clock::now();
        pack_func(&tvm_A, &tvm_packedA);
        // auto mid_time = std::chrono::steady_clock::now();
        // printf("packed_func execution done\n");
        func(&tvm_packedA, &tvm_B, &tvm_C);
        // auto end_time = std::chrono::steady_clock::now();
        // printf("packed_func and func execution done\n");

        // double pack_time_second = std::chrono::duration<double>(mid_time - start_time).count();
        // double matmul_time_second = std::chrono::duration<double>(end_time - mid_time).count();
        // std::cout << "Pack time: " << pack_time_second << " s " << "Matmul time: " << matmul_time_second << " s " << std::endl;

        free(packedA);
        // printf("Free packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));
    } else if (packAB == 2) {
        int packedB_size = it.packedB_size;
        // printf("Allocating packedB size = %lu Bytes\n", packedB_size * sizeof(float));
        float *packedB = static_cast<float*>(_mm_malloc(64, packedB_size * sizeof(float)));

        // printf("Begin allocating DLTensors\n");
        DLTensor tvm_packedB;
        tvm_packedB.device = device;
        tvm_packedB.dtype = dtype;
        tvm_packedB.ndim = 4;
        tvm_packedB.shape = it.packedB_shape;
        tvm_packedB.data = packedB;
        tvm_packedB.strides = nullptr;
        tvm_packedB.byte_offset = 0;
        tvm_C.byte_offset = 0;
        // printf("packed_func and func executing\n");
        // auto start_time = std::chrono::steady_clock::now();
        pack_func(&tvm_B, &tvm_packedB);
        // auto mid_time = std::chrono::steady_clock::now();
        // printf("packed_func execution done\n");
        func(&tvm_A, &tvm_packedB, &tvm_C);
        // auto end_time = std::chrono::steady_clock::now();
        // printf("packed_func and func execution done\n");

        // double pack_time_second = std::chrono::duration<double>(mid_time - start_time).count();
        // double matmul_time_second = std::chrono::duration<double>(end_time - mid_time).count();
        // std::cout << "Pack time: " << pack_time_second << " s " << "Matmul time: " << matmul_time_second << " s " << std::endl;
        free(packedB);
        // printf("Free packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));
    }
}

#endif