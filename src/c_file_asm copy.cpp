
#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#endif
#include <cmath>
#include <cstring>
#include <cassert>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdio>
#include "test.h"
#include "timer.h"

namespace laf {
void small_gemm(const float *A, const float *B, float *C, int lda, int ldb, int ldc) {
asm volatile(
"\n" // 进入了整个small_gemm的初始化...
    "prfm    PLDL1KEEP, [%[A], #64]     \n" // A矩阵预取
    "prfm    PLDL1KEEP, [%[B], #64]     \n" // B矩阵预取
    "lsl     %[lda], %[lda], #2             \n" // x6存储lda乘以FLOAT_BYTES，方便后面做偏移
    "lsl     %[ldb], %[ldb], #3             \n" // x8存储ldb乘以FLOAT_BYTES再乘以2（跳两行）
    "lsl     %[ldc], %[ldc], #2             \n" // x9存储ldc乘以FLOAT_BYTES
    "mov     x21, %[A]                  \n" // x21存储A头指针
    "mov     x24, %[C]                  \n" // x24存储C头指针
"\n" // 进入了整个small_gemm的初始化...完成
"\n" // 进入了N方向剩余循环...
    "ptrue     p0.s                  \n"
    "mov       x28, #5                  \n"
    "whilelt   p1.s, xzr, x28                  \n"
"\n" // 进入了N方向的初始化...
"\n" // 进入了x寄存器初始化阶段...
    "mov     x22, %[B]                   \n" // x11存储B头指针
    "add     x23, %[B], %[ldb], lsr #1               \n" // x12存储B + FLOAT_BYTES * ldb
    "prfm    PLDL1KEEP, [x22, #64]              \n" // B矩阵预取
    "prfm    PLDL1KEEP, [x23, #64]              \n" // B矩阵预取
"\n" // 进入了x寄存器初始化阶段...完成
"\n" // 进入了A矩阵x寄存器初始化...
    "mov     x12, x21    \n" // 将x21(A矩阵头指针)存入x12
    "add     x13, x21, %[lda]    \n" // 将x21加上%[lda]后存入x13
    "add     x14, x12, %[lda], lsl #1    \n"// 将x12加上2倍的%[lda]后存入x14
    "add     x15, x13, %[lda], lsl #1    \n"// 将x13加上2倍的%[lda]后存入x15
    "add     x16, x14, %[lda], lsl #1    \n"// 将x14加上2倍的%[lda]后存入x16
    "add     x17, x15, %[lda], lsl #1    \n"// 将x15加上2倍的%[lda]后存入x17
"\n" // 进入了A矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "mov     x6, x24    \n" // 将x24(C矩阵头指针)存入x6
    "add     x7, x24, %[ldc]     \n" // 将x24加上%[ldc]后存入x7
    "add     x8, x6, %[ldc], lsl #1    \n" // 将x6加上2倍的%[ldc]后存入x8
    "add     x9, x7, %[ldc], lsl #1    \n" // 将x7加上2倍的%[ldc]后存入x9
    "add     x10, x8, %[ldc], lsl #1    \n" // 将x8加上2倍的%[ldc]后存入x10
    "add     x11, x9, %[ldc], lsl #1    \n" // 将x9加上2倍的%[ldc]后存入x11
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
"\n" // 进入了N方向的初始化...完成
"\n" // 进入了N方向的剩余循环...
"\n" // 进入了M方向的主循环...
    "mov     x26, #2                   \n" // x26存储MR_MAIN_LOOPS的值
    "b       1f                                      \n" // 跳到1
  "2:                                 \n" // K方向的剩余操作
    "subs    x26, x26, #1                            \n"
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "beq     3f              \n" // 等于0的话跳到3, 下方是M方向的主要操作
    "prfm    PSTL1KEEP, [x6, #64]              \n" // 从x6预取C矩阵数据
    "prfm    PSTL1KEEP, [x7, #64]              \n" // 从x7预取C矩阵数据
    "prfm    PSTL1KEEP, [x8, #64]              \n" // 从x8预取C矩阵数据
    "prfm    PSTL1KEEP, [x9, #64]              \n" // 从x9预取C矩阵数据
    "prfm    PSTL1KEEP, [x10, #64]              \n" // 从x10预取C矩阵数据
    "prfm    PSTL1KEEP, [x11, #64]              \n" // 从x11预取C矩阵数据
    "st1w     z8.s, p0, [x6, #0, mul vl]           \n"
    "st1w     z9.s, p1, [x6, #1, mul vl]           \n"
    "st1w     z10.s, p0, [x7, #0, mul vl]           \n"
    "st1w     z11.s, p1, [x7, #1, mul vl]           \n"
    "st1w     z12.s, p0, [x8, #0, mul vl]           \n"
    "st1w     z13.s, p1, [x8, #1, mul vl]           \n"
    "st1w     z14.s, p0, [x9, #0, mul vl]           \n"
    "st1w     z15.s, p1, [x9, #1, mul vl]           \n"
    "st1w     z16.s, p0, [x10, #0, mul vl]           \n"
    "st1w     z17.s, p1, [x10, #1, mul vl]           \n"
    "st1w     z18.s, p0, [x11, #0, mul vl]           \n"
    "st1w     z19.s, p1, [x11, #1, mul vl]           \n"
"\n" // 进入了x寄存器初始化阶段...
    "mov     x22, %[B]                   \n" // x11存储B头指针
    "add     x23, %[B], %[ldb], lsr #1               \n" // x12存储B + FLOAT_BYTES * ldb
    "prfm    PLDL1KEEP, [x22, #64]              \n" // B矩阵预取
    "prfm    PLDL1KEEP, [x23, #64]              \n" // B矩阵预取
"\n" // 进入了x寄存器初始化阶段...完成
"\n" // 进入了A矩阵x寄存器初始化...
    "mov     x12, x21    \n" // 将x21(A矩阵头指针)存入x12
    "add     x13, x21, %[lda]    \n" // 将x21加上%[lda]后存入x13
    "add     x14, x12, %[lda], lsl #1    \n"// 将x12加上2倍的%[lda]后存入x14
    "add     x15, x13, %[lda], lsl #1    \n"// 将x13加上2倍的%[lda]后存入x15
    "add     x16, x14, %[lda], lsl #1    \n"// 将x14加上2倍的%[lda]后存入x16
    "add     x17, x15, %[lda], lsl #1    \n"// 将x15加上2倍的%[lda]后存入x17
"\n" // 进入了A矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "mov     x6, x24    \n" // 将x24(C矩阵头指针)存入x6
    "add     x7, x24, %[ldc]     \n" // 将x24加上%[ldc]后存入x7
    "add     x8, x6, %[ldc], lsl #1    \n" // 将x6加上2倍的%[ldc]后存入x8
    "add     x9, x7, %[ldc], lsl #1    \n" // 将x7加上2倍的%[ldc]后存入x9
    "add     x10, x8, %[ldc], lsl #1    \n" // 将x8加上2倍的%[ldc]后存入x10
    "add     x11, x9, %[ldc], lsl #1    \n" // 将x9加上2倍的%[ldc]后存入x11
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
  "1:                                 \n" // K方向的主要操作
    "mov     x27, #2                   \n" // x27存储K方向的循环次数Main_K_loop_times
    "subs    x27, x27, #1                            \n"
"\n" // 进入了K方向的初始化...
    "add     x21, x21, %[lda], lsl #1               \n" // A矩阵头指针加上x6*2^1倍
    "add     x24, x24, %[ldc], lsl #1               \n" // C矩阵头指针加上x9*2^1倍
    "add     x21, x21, %[lda], lsl #2               \n" // A矩阵头指针加上x6*2^2倍
    "add     x24, x24, %[ldc], lsl #2               \n" // C矩阵头指针加上x9*2^2倍
    "fmul    z8.s, z6.s, z0.s             \n"
    "fmul    z9.s, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmul    z10.s, z6.s, z1.s             \n"
    "fmul    z11.s, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmul    z12.s, z6.s, z2.s             \n"
    "fmul    z13.s, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmul    z14.s, z6.s, z3.s             \n"
    "fmul    z15.s, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmul    z16.s, z6.s, z4.s             \n"
    "fmul    z17.s, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmul    z18.s, z6.s, z5.s             \n"
    "fmul    z19.s, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
"\n" // 进入了K方向的初始化...完成
    "b       4f                                 \n" // 跳到4
  "5:                                 \n"
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
  "4:                                 \n"
    "beq     2b                       \n" // 等于0的话跳到2
    "subs    x27, x27, #1                            \n" // Main_K_loop_times -= 1
    "b       5b                                 \n" // 跳转到5
  "3:                                 \n"
"\n" // 进入了M方向的主循环...完成
"\n" // 进入了N方向的剩余循环...完成
    "prfm    PSTL1KEEP, [x6, #64]              \n" // 从x6预取C矩阵数据
    "prfm    PSTL1KEEP, [x7, #64]              \n" // 从x7预取C矩阵数据
    "prfm    PSTL1KEEP, [x8, #64]              \n" // 从x8预取C矩阵数据
    "prfm    PSTL1KEEP, [x9, #64]              \n" // 从x9预取C矩阵数据
    "prfm    PSTL1KEEP, [x10, #64]              \n" // 从x10预取C矩阵数据
    "prfm    PSTL1KEEP, [x11, #64]              \n" // 从x11预取C矩阵数据
    "st1w     z8.s, p0, [x6, #0, mul vl]           \n"
    "st1w     z9.s, p1, [x6, #1, mul vl]           \n"
    "st1w     z10.s, p0, [x7, #0, mul vl]           \n"
    "st1w     z11.s, p1, [x7, #1, mul vl]           \n"
    "st1w     z12.s, p0, [x8, #0, mul vl]           \n"
    "st1w     z13.s, p1, [x8, #1, mul vl]           \n"
    "st1w     z14.s, p0, [x9, #0, mul vl]           \n"
    "st1w     z15.s, p1, [x9, #1, mul vl]           \n"
    "st1w     z16.s, p0, [x10, #0, mul vl]           \n"
    "st1w     z17.s, p1, [x10, #1, mul vl]           \n"
    "st1w     z18.s, p0, [x11, #0, mul vl]           \n"
    "st1w     z19.s, p1, [x11, #1, mul vl]           \n"
"\n" // 进入了N方向剩余循环...完成
: [A]"=r"(A),
  [B]"=r"(B),
  [C]"=r"(C),
  [lda]"=r"(lda),
  [ldb]"=r"(ldb),
  [ldc]"=r"(ldc) 
: "0"(A),
  "1"(B),
  "2"(C),
  "3"(lda),
  "4"(ldb),
  "5"(ldc) 
: "cc", "memory"
  , "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
, "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );

}
void small_gemm_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc) {
asm volatile(
"\n" // 进入了整个small_gemm的初始化...
    "prfm    PLDL1KEEP, [%[A], #64]     \n" // A矩阵预取
    "prfm    PLDL1KEEP, [%[B], #64]     \n" // B矩阵预取
    "lsl     %[lda], %[lda], #2             \n" // x6存储lda乘以FLOAT_BYTES，方便后面做偏移
    "lsl     %[ldb], %[ldb], #3             \n" // x8存储ldb乘以FLOAT_BYTES再乘以2（跳两行）
    "lsl     %[ldc], %[ldc], #2             \n" // x9存储ldc乘以FLOAT_BYTES
    "mov     x21, %[A]                  \n" // x21存储A头指针
    "mov     x24, %[C]                  \n" // x24存储C头指针
"\n" // 进入了整个small_gemm的初始化...完成
"\n" // 进入了N方向剩余循环...
    "ptrue     p0.s                  \n"
    "mov       x28, #5                  \n"
    "whilelt   p1.s, xzr, x28                  \n"
"\n" // 进入了N方向的初始化...
"\n" // 进入了x寄存器初始化阶段...
    "mov     x22, %[B]                   \n" // x11存储B头指针
    "add     x23, %[B], %[ldb], lsr #1               \n" // x12存储B + FLOAT_BYTES * ldb
    "prfm    PLDL1KEEP, [x22, #64]              \n" // B矩阵预取
    "prfm    PLDL1KEEP, [x23, #64]              \n" // B矩阵预取
"\n" // 进入了x寄存器初始化阶段...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "mov     x6, x24    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z8.s, p0/z, [x6, #0, mul vl]           \n"
    "ld1w     z9.s, p0/z, [x6, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵x寄存器初始化...
    "mov     x12, x21    \n" // 将x21(A矩阵头指针)存入x12
    "add     x13, x21, %[lda]    \n" // 将x21加上%[lda]后存入x13
    "add     x14, x12, %[lda], lsl #1    \n"// 将x12加上2倍的%[lda]后存入x14
    "add     x15, x13, %[lda], lsl #1    \n"// 将x13加上2倍的%[lda]后存入x15
    "add     x16, x14, %[lda], lsl #1    \n"// 将x14加上2倍的%[lda]后存入x16
    "add     x17, x15, %[lda], lsl #1    \n"// 将x15加上2倍的%[lda]后存入x17
"\n" // 进入了A矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x7, x24, %[ldc]     \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z10.s, p0/z, [x7, #0, mul vl]           \n"
    "ld1w     z11.s, p0/z, [x7, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x8, x6, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z12.s, p0/z, [x8, #0, mul vl]           \n"
    "ld1w     z13.s, p0/z, [x8, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x9, x7, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z14.s, p0/z, [x9, #0, mul vl]           \n"
    "ld1w     z15.s, p0/z, [x9, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x10, x8, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z16.s, p0/z, [x10, #0, mul vl]           \n"
    "ld1w     z17.s, p0/z, [x10, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x11, x9, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z18.s, p0/z, [x11, #0, mul vl]           \n"
    "ld1w     z19.s, p0/z, [x11, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "ld1rw     z0.s, p0/z, [x12]    \n"
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
    "ld1rw     z1.s, p0/z, [x13]    \n"
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
    "ld1rw     z2.s, p0/z, [x14]    \n"
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
    "ld1rw     z3.s, p0/z, [x15]    \n"
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
    "ld1rw     z4.s, p0/z, [x16]    \n"
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
    "ld1rw     z5.s, p0/z, [x17]    \n"
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了N方向的初始化...完成
"\n" // 进入了N方向的剩余循环...
"\n" // 进入了M方向的主循环...
    "mov     x26, #2                   \n" // x26存储MR_MAIN_LOOPS的值
    "b       1f                                      \n" // 跳到1
  "2:                                 \n" // K方向的剩余操作
    "subs    x26, x26, #1                            \n"
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "beq     3f              \n" // 等于0的话跳到3, 下方是M方向的主要操作
    "prfm    PSTL1KEEP, [x6, #64]              \n" // 从x6预取C矩阵数据
    "prfm    PSTL1KEEP, [x7, #64]              \n" // 从x7预取C矩阵数据
    "prfm    PSTL1KEEP, [x8, #64]              \n" // 从x8预取C矩阵数据
    "prfm    PSTL1KEEP, [x9, #64]              \n" // 从x9预取C矩阵数据
    "prfm    PSTL1KEEP, [x10, #64]              \n" // 从x10预取C矩阵数据
    "prfm    PSTL1KEEP, [x11, #64]              \n" // 从x11预取C矩阵数据
    "st1w     z8.s, p0, [x6, #0, mul vl]           \n"
    "st1w     z9.s, p1, [x6, #1, mul vl]           \n"
    "st1w     z10.s, p0, [x7, #0, mul vl]           \n"
    "st1w     z11.s, p1, [x7, #1, mul vl]           \n"
    "st1w     z12.s, p0, [x8, #0, mul vl]           \n"
    "st1w     z13.s, p1, [x8, #1, mul vl]           \n"
    "st1w     z14.s, p0, [x9, #0, mul vl]           \n"
    "st1w     z15.s, p1, [x9, #1, mul vl]           \n"
    "st1w     z16.s, p0, [x10, #0, mul vl]           \n"
    "st1w     z17.s, p1, [x10, #1, mul vl]           \n"
    "st1w     z18.s, p0, [x11, #0, mul vl]           \n"
    "st1w     z19.s, p1, [x11, #1, mul vl]           \n"
"\n" // 进入了x寄存器初始化阶段...
    "mov     x22, %[B]                   \n" // x11存储B头指针
    "add     x23, %[B], %[ldb], lsr #1               \n" // x12存储B + FLOAT_BYTES * ldb
    "prfm    PLDL1KEEP, [x22, #64]              \n" // B矩阵预取
    "prfm    PLDL1KEEP, [x23, #64]              \n" // B矩阵预取
"\n" // 进入了x寄存器初始化阶段...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "mov     x6, x24    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z8.s, p0/z, [x6, #0, mul vl]           \n"
    "ld1w     z9.s, p0/z, [x6, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵x寄存器初始化...
    "mov     x12, x21    \n" // 将x21(A矩阵头指针)存入x12
    "add     x13, x21, %[lda]    \n" // 将x21加上%[lda]后存入x13
    "add     x14, x12, %[lda], lsl #1    \n"// 将x12加上2倍的%[lda]后存入x14
    "add     x15, x13, %[lda], lsl #1    \n"// 将x13加上2倍的%[lda]后存入x15
    "add     x16, x14, %[lda], lsl #1    \n"// 将x14加上2倍的%[lda]后存入x16
    "add     x17, x15, %[lda], lsl #1    \n"// 将x15加上2倍的%[lda]后存入x17
"\n" // 进入了A矩阵x寄存器初始化...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x7, x24, %[ldc]     \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z10.s, p0/z, [x7, #0, mul vl]           \n"
    "ld1w     z11.s, p0/z, [x7, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x8, x6, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z12.s, p0/z, [x8, #0, mul vl]           \n"
    "ld1w     z13.s, p0/z, [x8, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x9, x7, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z14.s, p0/z, [x9, #0, mul vl]           \n"
    "ld1w     z15.s, p0/z, [x9, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x10, x8, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z16.s, p0/z, [x10, #0, mul vl]           \n"
    "ld1w     z17.s, p0/z, [x10, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了C矩阵x寄存器初始化...
    "add     x11, x9, %[ldc], lsl #1    \n"
"\n" // 进入了C矩阵x寄存器初始化...完成
"\n" // 进入了C矩阵数据加载...
    "ld1w     z18.s, p0/z, [x11, #0, mul vl]           \n"
    "ld1w     z19.s, p0/z, [x11, #1, mul vl]           \n"
"\n" // 进入了C矩阵数据加载...完成
"\n" // 进入了A矩阵数据加载...
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "ld1rw     z0.s, p0/z, [x12]    \n"
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
    "ld1rw     z1.s, p0/z, [x13]    \n"
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
    "ld1rw     z2.s, p0/z, [x14]    \n"
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
    "ld1rw     z3.s, p0/z, [x15]    \n"
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
    "ld1rw     z4.s, p0/z, [x16]    \n"
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
    "ld1rw     z5.s, p0/z, [x17]    \n"
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
  "1:                                 \n" // K方向的主要操作
    "mov     x27, #2                   \n" // x27存储K方向的循环次数Main_K_loop_times
    "subs    x27, x27, #1                            \n"
"\n" // 进入了K方向的初始化...
    "add     x21, x21, %[lda], lsl #1               \n" // A矩阵头指针加上x6*2^1倍
    "add     x24, x24, %[ldc], lsl #1               \n" // C矩阵头指针加上x9*2^1倍
    "add     x21, x21, %[lda], lsl #2               \n" // A矩阵头指针加上x6*2^2倍
    "add     x24, x24, %[ldc], lsl #2               \n" // C矩阵头指针加上x9*2^2倍
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
"\n" // 进入了K方向的初始化...完成
    "b       4f                                 \n" // 跳到4
  "5:                                 \n"
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x22, #0, mul vl]             \n" // 将x22 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x22, #1, mul vl]             \n" // 将x22 + #32 Bytes处的数据加载到q7当中
    "add     x22, x22, %[ldb]              \n" // 将x22加上%[ldb]后存入x22
"\n" // 进入了B矩阵数据加载...完成
    "fmla    z8.s, p0/m, z6.s, z0.s             \n"
    "fmla    z9.s, p1/m, z7.s, z0.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z0.s, p0/z, [x12]    \n"// 将x12处的数据加载到q0中
    "add     x12, x12, #4    \n"// 使x12偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z10.s, p0/m, z6.s, z1.s             \n"
    "fmla    z11.s, p1/m, z7.s, z1.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z1.s, p0/z, [x13]    \n"// 将x13处的数据加载到q1中
    "add     x13, x13, #4    \n"// 使x13偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z12.s, p0/m, z6.s, z2.s             \n"
    "fmla    z13.s, p1/m, z7.s, z2.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z2.s, p0/z, [x14]    \n"// 将x14处的数据加载到q2中
    "add     x14, x14, #4    \n"// 使x14偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z14.s, p0/m, z6.s, z3.s             \n"
    "fmla    z15.s, p1/m, z7.s, z3.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z3.s, p0/z, [x15]    \n"// 将x15处的数据加载到q3中
    "add     x15, x15, #4    \n"// 使x15偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z16.s, p0/m, z6.s, z4.s             \n"
    "fmla    z17.s, p1/m, z7.s, z4.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z4.s, p0/z, [x16]    \n"// 将x16处的数据加载到q4中
    "add     x16, x16, #4    \n"// 使x16偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
    "fmla    z18.s, p0/m, z6.s, z5.s             \n"
    "fmla    z19.s, p1/m, z7.s, z5.s             \n"
"\n" // 进入了A矩阵数据加载...
    "ld1rw     z5.s, p0/z, [x17]    \n"// 将x17处的数据加载到q5中
    "add     x17, x17, #4    \n"// 使x17偏移SIMD的长度
"\n" // 进入了A矩阵数据加载...完成
"\n" // 进入了B矩阵数据加载...
    "ld1w     z6.s, p0/z, [x23, #0, mul vl]             \n" // 将x23 + #0 Bytes处的数据加载到q6当中
    "ld1w     z7.s, p0/z, [x23, #1, mul vl]             \n" // 将x23 + #32 Bytes处的数据加载到q7当中
    "add     x23, x23, %[ldb]              \n" // 将x23加上%[ldb]后存入x23
"\n" // 进入了B矩阵数据加载...完成
  "4:                                 \n"
    "beq     2b                       \n" // 等于0的话跳到2
    "subs    x27, x27, #1                            \n" // Main_K_loop_times -= 1
    "b       5b                                 \n" // 跳转到5
  "3:                                 \n"
"\n" // 进入了M方向的主循环...完成
"\n" // 进入了N方向的剩余循环...完成
    "prfm    PSTL1KEEP, [x6, #64]              \n" // 从x6预取C矩阵数据
    "prfm    PSTL1KEEP, [x7, #64]              \n" // 从x7预取C矩阵数据
    "prfm    PSTL1KEEP, [x8, #64]              \n" // 从x8预取C矩阵数据
    "prfm    PSTL1KEEP, [x9, #64]              \n" // 从x9预取C矩阵数据
    "prfm    PSTL1KEEP, [x10, #64]              \n" // 从x10预取C矩阵数据
    "prfm    PSTL1KEEP, [x11, #64]              \n" // 从x11预取C矩阵数据
    "st1w     z8.s, p0, [x6, #0, mul vl]           \n"
    "st1w     z9.s, p1, [x6, #1, mul vl]           \n"
    "st1w     z10.s, p0, [x7, #0, mul vl]           \n"
    "st1w     z11.s, p1, [x7, #1, mul vl]           \n"
    "st1w     z12.s, p0, [x8, #0, mul vl]           \n"
    "st1w     z13.s, p1, [x8, #1, mul vl]           \n"
    "st1w     z14.s, p0, [x9, #0, mul vl]           \n"
    "st1w     z15.s, p1, [x9, #1, mul vl]           \n"
    "st1w     z16.s, p0, [x10, #0, mul vl]           \n"
    "st1w     z17.s, p1, [x10, #1, mul vl]           \n"
    "st1w     z18.s, p0, [x11, #0, mul vl]           \n"
    "st1w     z19.s, p1, [x11, #1, mul vl]           \n"
"\n" // 进入了N方向剩余循环...完成
: [A]"=r"(A),
  [B]"=r"(B),
  [C]"=r"(C),
  [lda]"=r"(lda),
  [ldb]"=r"(ldb),
  [ldc]"=r"(ldc) 
: "0"(A),
  "1"(B),
  "2"(C),
  "3"(lda),
  "4"(ldb),
  "5"(ldc) 
: "cc", "memory"
  , "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
, "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );

}
}

extern "C" int gemm_12x11x13_11_13_13_xsmm_VCMGFSNH(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc){
  laf::small_gemm(A, B, C, lda, ldb, ldc);
  return 0;
}

extern "C" int gemm_12x11x13_11_13_13_xsmm_with_bias_VCMGFSNH(const float *A, const float *B, float *C, const int lda, const int ldb, const int ldc){
  laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  return 0;
}

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

int main() {
  #define M 12
  #define N 13
  #define K 11

  #define lda 11
  #define ldb 13
  #define ldc 13

  float *A = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  float *B = static_cast<float*>(_mm_malloc(64, K * ldb * sizeof(float)));
  float *C = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *refC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *ourC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));

  test_utils::init(A, M * lda);
  test_utils::init(B, K * ldb);
  test_utils::init(C, M * ldc);

  int n_warming = 20;
  int n_loops = 38191142;

  for (int i = 0; i < n_warming; ++i) {
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }

  Timer t;
  for (int i = 0; i < n_loops; ++i) {
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }

  float latency = t.getTime();
  float gflops = M * N * K * 2 / latency * n_loops / 1000000;
  printf("%.2f, ", gflops);

  bool ACC = false;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  laf::small_gemm(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-4, 1e-4)) {
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-4, 1e-4);
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
    test_utils::print_diff(refC, ourC, M, N, ldc);
  } else {
    //printf("0------passed\n");
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = 10.0f * rand() / RAND_MAX;
      refC[i * ldc + j] = c;
      ourC[i * ldc + j] = c;
    }
  }
  ACC = true;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  laf::small_gemm_with_bias(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, 1e-4, 1e-4)) {
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, 1e-4, 1e-4);
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
    test_utils::print_diff(refC, ourC, M, N, ldc);
  } else {
    //printf("1------passed\n");
  }


  free(A);
  A=NULL;
  free(B);
  B=NULL;
  free(C);
  C=NULL;
  free(refC);
  refC=NULL;
  free(ourC);
  ourC=NULL;
}
