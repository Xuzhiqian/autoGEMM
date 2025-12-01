#ifndef __AUTOGEMM_SGEMM__
#define __AUTOGEMM_SGEMM__

typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113, // conjugate transpose
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

typedef int BLASINT;

void autogemm_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
    const BLASINT m, const BLASINT n, const BLASINT k, const float alpha, const float *a, const BLASINT lda,
    const float *b, const BLASINT ldb, const float beta, float *c, const BLASINT ldc);

#endif /* __AUTOGEMM_SGEMM__ */