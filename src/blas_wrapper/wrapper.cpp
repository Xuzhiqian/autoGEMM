#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
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

    int nc, kc, padding_size;
    KernelParams::CreateList();
    for(auto it = KernelParams::params_list.begin(); it != KernelParams::params_list.end(); it++){
      if (it->M == M && it->N == N && it->K == K) {
        nc = it->nc;
        kc = it->kc;
        padding_size = it->padding_size;
        break;
      }
    }

    int nc_ceil = ((nc - 1) / padding_size + 1) * padding_size;

    // printf("m = %d, n = %d, k = %d, nc = %d, kc = %d, padding_size = %d， nc_ceil = %d\n", m, n, k, nc, kc, padding_size, nc_ceil);
    // printf("Allocating packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));

    float *packedB = static_cast<float*>(_mm_malloc(64, K * (N/nc) * nc_ceil * sizeof(float)));

    std::string base_name = "GEMM_" + std::to_string(M) + "X" + std::to_string(N) + "X" + std::to_string(K);
    std::string mod_name = base_name + "_kernel.so";
    std::string func_name = "OP_" + base_name;
    std::string pack_func_name = func_name + "_packB";

    // std::cout << std::filesystem::current_path() << std::endl;

    tvm::runtime::Module mod_tvmlib = tvm::runtime::Module::LoadFromFile("/home/linzuxuan/tensorflow-ci/tensorflow/third_party/autogemm/data/tune_output/build/library/" + mod_name);
    tvm::runtime::PackedFunc pack_func = mod_tvmlib.GetFunction(pack_func_name);
    tvm::runtime::PackedFunc func = mod_tvmlib.GetFunction(func_name);

    // printf("Begin allocating DLTensors\n");
    DLTensor tvm_A;
    DLTensor tvm_B;
    DLTensor tvm_packedB;
    DLTensor tvm_C;

    int64_t A_shape[] = {M, K};
    int64_t B_shape[] = {K, N};
    int64_t packedB_shape[] = {K / kc, N / nc, kc, nc_ceil};
    int64_t C_shape[] = {M, N};

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    // TVMArrayAlloc(A_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_A);
    // TVMArrayAlloc(B_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_B);
    // TVMArrayAlloc(packedB_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_packedB);
    // TVMArrayAlloc(C_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_C);

    // printf("tvm_A.data = %p, tvm_B.data = %p, tvm_packedB.data = %p, tvm_C.data = %p\n", tvm_A.data, tvm_B.data, tvm_packedB.data, tvm_C.data);

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
    tvm_A.shape = A_shape;
    tvm_B.shape = B_shape;
    tvm_packedB.shape = packedB_shape;
    tvm_C.shape = C_shape;

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

    // printf("packed_func and func executing\n");
    pack_func(&tvm_B, &tvm_packedB);
    func(&tvm_A, &tvm_packedB, &tvm_C);
    // printf("packed_func and func execution done\n");

    // pack_func(b, packedB);
    // func(a, packedB, c);

    free(packedB);
    // printf("Free packedB size = %lu Bytes\n", K * (N/nc) * nc_ceil * sizeof(float));

    // TVMArrayFree(tvm_A);
    // TVMArrayFree(tvm_B);
    // TVMArrayFree(tvm_packedB);
    // TVMArrayFree(tvm_C);
}

#endif