#ifndef __TVMGEMM_UTIL_H_
#define __TVMGEMM_UTIL_H_

#include <cstdio>
#include <cstdlib>

#include "dlpack/dlpack.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

#include "../../../src/blas_wrapper/include/kernel_params_list.hpp"
#include "./test.h"
#include "./timer.h"

void *_mm_malloc(size_t align, size_t sz)
{
    void *ptr;
    int alloc_result = posix_memalign(&ptr, align, sz);
    if (alloc_result != 0) {
        return NULL;
    }
    return ptr;
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int repeat = atoi(argv[4]);

    KernelParams::CreateMap();
    std::string query_key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    auto it = KernelParams::mapping.find(query_key)->second;

    tvm::runtime::PackedFunc pack_func = it.pack_func;
    tvm::runtime::PackedFunc func = it.func;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    float *A = static_cast<float *>(_mm_malloc(64, M * lda * sizeof(float)));
    float *B = static_cast<float *>(_mm_malloc(64, K * ldb * sizeof(float)));
    float *C = static_cast<float *>(_mm_malloc(64, M * ldc * sizeof(float)));
    float *refC = static_cast<float *>(_mm_malloc(64, M * ldc * sizeof(float)));
    float *ourC = C;

    test_utils::init(A, M * lda);
    test_utils::init(B, K * ldb);
    test_utils::init(C, M * ldc);

    DLTensor *tvm_A;
    DLTensor *tvm_B;
    DLTensor *tvm_C;

    const int dtype_code = kDLFloat;
    const int dtype_bits = 32;
    const int dtype_lanes = 1;
    const int device_type = kDLCPU;
    const int device_id = 0;

    TVMArrayAlloc(it.A_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_A);
    TVMArrayAlloc(it.B_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_B);
    TVMArrayAlloc(it.C_shape, 2, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_C);

    tvm_A->data = A;
    tvm_B->data = B;
    tvm_C->data = C;

    double alpha = 1.0;
    double beta = 0.0;
    int n_warming = 20;
    int n_loops = repeat;

    float latency_offline = 0.0;
    float latency_online = 0.0;

    bool ACC = false;
    test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);

    int packAB = it.packAB;
    if (packAB == 0 || packAB == 2) {
        // warming
        for (int i = 0; i < n_warming; ++i) {
            func(tvm_A, tvm_B, tvm_C);
        }

        // Testing performance
        Timer t_1;
        for (int i = 0; i < n_loops; ++i) {
            func(tvm_A, tvm_B, tvm_C);
        }
        latency_offline = t_1.getTime();

        Timer t_2;
        for (int i = 0; i < n_loops; ++i) {
            func(tvm_A, tvm_B, tvm_C);
        }
        latency_online = t_2.getTime();

        // Test accuracy
        func(tvm_A, tvm_B, tvm_C);
    } else if (packAB == 1) {
        int packedA_size = it.packedA_size;
        float *packedA = static_cast<float *>(_mm_malloc(64, packedA_size * sizeof(float)));

        DLTensor *tvm_packedA;
        TVMArrayAlloc(it.packedA_shape, 4, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &tvm_packedA);
        tvm_packedA->data = packedA;

        // warming
        pack_func(tvm_A, tvm_packedA);
        for (int i = 0; i < n_warming; ++i) {
            func(tvm_packedA, tvm_B, tvm_C);
        }

        // Testing performance
        Timer t_1;
        for (int i = 0; i < n_loops; ++i) {
            func(tvm_packedA, tvm_B, tvm_C);
        }
        latency_offline = t_1.getTime();

        Timer t_2;
        for (int i = 0; i < n_loops; ++i) {
            pack_func(tvm_A, tvm_packedA);
            func(tvm_packedA, tvm_B, tvm_C);
        }
        latency_online = t_2.getTime();

        // Test accuracy
        pack_func(tvm_A, tvm_packedA);
        func(tvm_packedA, tvm_B, tvm_C);

        free(packedA);
    }

    float gflops = M * N * K / latency_offline / 1000000 * n_loops * 2;
    printf("offline, M: %d, N: %d, K: %d, perf: %.2f gflops, latency: %.6f ms\n", M, N, K, gflops, latency_offline / n_loops);

    gflops = M * N * K / latency_online / 1000000 * n_loops * 2;
    printf("online, M: %d, N: %d, K: %d, perf: %.2f gflops, latency: %.6f ms\n", M, N, K, gflops, latency_online / n_loops);

    if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-5, 1e-5)) {
        int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-5, 1e-5);
        printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
            M, N, K,
            lda, ldb, ldc,
            ACC,
            idx, refC[idx],
            idx, ourC[idx]
        );
    } else {
        // printf("0------passed\n");
    }

    free(A);
    free(B);
    free(C);
    free(refC);

    return 0;
}

#endif