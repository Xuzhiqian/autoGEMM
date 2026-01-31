#!/bin/bash
set -e

# parameter setting
M=32
N=32
K=32
lda=32
ldb=32
ldc=32
UNROLL_K=4
NR_MAIN=8
REPEAT=64
PIPELINE_STRATEGY_LEVEL=0
MSRA_FLAG=0

for arg in "$@"; do
    shift
    case "$arg" in
        --M) M="$1" ;;
        --N) N="$1" ;;
        --K) K="$1" ;;
        --lda) lda="$1" ;;
        --ldb) ldb="$1" ;;
        --ldc) ldc="$1" ;;
        --UNROLL_K) UNROLL_K="$1" ;;
        --NR_MAIN) NR_MAIN="$1" ;;
        --REPEAT) REPEAT="$1" ;;
        --PIPELINE_STRATEGY_LEVEL) PIPELINE_STRATEGY_LEVEL="$1" ;;
        --MSRA_FLAG) MSRA_FLAG="$1" ;;
    esac
done

# directory setting
CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
PROJECT_ROOT=`cd $CURRENT_PATH/../..; pwd`
# cd ${PROJECT_ROOT}
# echo "CURRENT_PATH: $CURRENT_PATH"
# echo "Project root: $PROJECT_ROOT"
SRC_PATH=$PROJECT_ROOT/src

export PYTHONPATH=$PYTHONPATH:$SRC_PATH:$SRC_PATH/tvm_tuner:$SRC_PATH/micro_kernel
export TVM_CC=clang++

echo -n "M=$M, N=$N, K=$K, lda=$lda, ldb=$ldb, ldc=$ldc, UNROLL_K=$UNROLL_K, NR_MAIN=$NR_MAIN, REPEAT=$REPEAT, PIPELINE_STRATEGY_LEVEL=$PIPELINE_STRATEGY_LEVEL, MSRA_FLAG=$MSRA_FLAG "

python $SRC_PATH/micro_kernel/make_c_file_asm_pipeline_experiment.py $M $N $K $lda $ldb $ldc $UNROLL_K $NR_MAIN $REPEAT $PIPELINE_STRATEGY_LEVEL $MSRA_FLAG # generate c_file_asm.cpp and Makefile into test/micro_kernel_tester
cd ${CURRENT_PATH}
make -s
./benchmark_kernel

echo ""
