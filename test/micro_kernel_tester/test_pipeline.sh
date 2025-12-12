#!/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
PROJECT_ROOT=`cd $CURRENT_PATH/../..; pwd`
cd ${PROJECT_ROOT}
echo "CURRENT_PATH: $CURRENT_PATH"
echo "Project root: $PROJECT_ROOT"
src_path=$PROJECT_ROOT/src

M=$1
N=$2
K=$3
lda=$4
ldb=$5
ldc=$6
UNROLL_K=$7
NR_MAIN=$8
MSRA_FLAG=$9

TOT_REPEAT=65536000000
REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
if test $REPEAT -gt 1000000000
then
	REPEAT=1000000000
fi

export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

echo -n $M, $N, $K, $lda, $ldb, $ldc, " "

python $src_path/micro_kernel/make_c_file_asm_pipeline_experiment.py $M $N $K $lda $ldb $ldc $UNROLL_K $NR_MAIN $REPEAT 0 $MSRA_FLAG
cd ${CURRENT_PATH}
make -s
./benchmark_kernel

python $src_path/micro_kernel/make_c_file_asm_pipeline_experiment.py $M $N $K $lda $ldb $ldc $UNROLL_K $NR_MAIN $REPEAT 1 $MSRA_FLAG
cd ${CURRENT_PATH}
make -s
./benchmark_kernel

python $src_path/micro_kernel/make_c_file_asm_pipeline_experiment.py $M $N $K $lda $ldb $ldc $UNROLL_K $NR_MAIN $REPEAT 2 $MSRA_FLAG
cd ${CURRENT_PATH}
make -s
./benchmark_kernel

python $src_path/micro_kernel/make_c_file_asm_pipeline_experiment.py $M $N $K $lda $ldb $ldc $UNROLL_K $NR_MAIN $REPEAT 3 $MSRA_FLAG
cd ${CURRENT_PATH}
make -s
./benchmark_kernel

echo ""
