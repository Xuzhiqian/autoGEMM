#!/bin/bash
set -e

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
tune_output_path=$PROJECT_ROOT/data/tune_output
src_path=$PROJECT_ROOT/src
build_output_path=$tune_output_path/build

# parameter setting
arch=$1
threads=$2
repeats=$3

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

if [[ -d $build_output_path ]]; then
    rm -rf $build_output_path
fi
mkdir -p $build_output_path
mkdir -p $build_output_path/gemm_obj
mkdir -p $build_output_path/generated_micro_kernel
mkdir -p $build_output_path/library

MNK_file=${tune_output_path}/MNK.txt 
scheduler_log=${tune_output_path}/scheduler_summary.log
scheduler_log_output=$build_output_path/scheduler_summary.log

export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

summarize_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/summarize_scheduler.py
build_kernel_params_list_path=$PROJECT_ROOT/src/tvm_tuner/build_kernel_params_list.py
python $summarize_scheduler_path --input $scheduler_log --output $scheduler_log_output
python $build_kernel_params_list_path

tester_src_path=$PROJECT_ROOT/test/tvm_tuner_tester/src
cd $src_path
make -s

evaluate_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/evaluate_scheduler.py
benchmark_kernel_path=$tester_src_path/benchmark_kernel
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python $evaluate_scheduler -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} --scheduler_log ${PROJECT_ROOT}/${scheduler_log_output}
    $benchmark_kernel_path ${M} ${N} ${K} ${repeats}
    let cnt+=1
done

touch $build_output_path/build.over
