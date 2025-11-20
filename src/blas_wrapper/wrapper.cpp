#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "dlpack/dlpack.h"
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

    // printf("nc = %d, kc = %d, padding_size = %d", nc, kc, padding_size);

    int nc_ceil = ((nc - 1) / padding_size + 1) * padding_size;

    float *packedB = static_cast<float*>(_mm_malloc(64, K * (N/nc) * nc_ceil * sizeof(float)));

    std::string base_name = "GEMM_" + std::to_string(M) + "X" + std::to_string(N) + "X" + std::to_string(K);
    std::string mod_name = base_name + "_kernel.so";
    std::string func_name = "OP_" + base_name;
    std::string pack_func_name = func_name + "_packB";

    // std::cout << std::filesystem::current_path() << std::endl;

    tvm::runtime::Module mod_tvmlib = tvm::runtime::Module::LoadFromFile("/home/linzuxuan/tensorflow-ci/tensorflow/third_party/autogemm/data/tune_output/build/library/" + mod_name);
    tvm::runtime::PackedFunc pack_func = mod_tvmlib.GetFunction(pack_func_name);
    tvm::runtime::PackedFunc func = mod_tvmlib.GetFunction(func_name);

    DLTensor*   tvm_A;
    DLTensor*   tvm_B;
    DLTensor*   tvm_packedB;
    DLTensor*   tvm_C;

    const int64_t A_shape[2] = {M, K};
    const int64_t B_shape[2] = {K, N};
    const int64_t packedB_shape[4] = {K / kc, N / nc, kc, nc_ceil};
    const int64_t C_shape[2] = {M, N};

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    TVMArrayAlloc(A_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_A);
    TVMArrayAlloc(B_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_B);
    TVMArrayAlloc(packedB_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_packedB);
    TVMArrayAlloc(C_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_C);

    tvm_A->data = (void *)a;
    tvm_B->data = (void *)b;
    tvm_packedB->data = packedB;
    tvm_C->data = (void *)c;

    pack_func(tvm_B, tvm_packedB);
    func(tvm_A, tvm_packedB, tvm_C);

    free(packedB);
}

#endif