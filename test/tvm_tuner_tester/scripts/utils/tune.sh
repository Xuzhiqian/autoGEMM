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
tune_num=$3

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

mkdir -p $tune_output_path
if [[ -d "$tune_output_path/build" ]]; then
    rm -rf $tune_output_path/build
fi
if [[ -d "$tune_output_path/perf" ]]; then
    rm -rf $tune_output_path/perf
fi
if [[ -d "$tune_output_path/log" ]]; then
    rm -rf $tune_output_path/log
fi
if [[ -f "$tune_output_path/scheduler_summary.log" ]]; then
    rm -rf $tune_output_path/scheduler_summary.log
fi
if [[ -f "$tune_output_path/tune.over" ]]; then
    rm -rf $tune_output_path/tune.over
fi
mkdir -p $tune_output_path/build
mkdir -p $tune_output_path/build/gemm_obj
mkdir -p $tune_output_path/build/generated_micro_kernel
mkdir -p $tune_output_path/build/library
mkdir -p $tune_output_path/perf
mkdir -p $tune_output_path/log

touch $tune_output_path/scheduler_summary.log

export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

MNK_file=${tune_output_path}/MNK.txt 
cnt=0
tune_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/tune_scheduler.py
matmul_log_path=$tune_output_path/matmul.log
matmul_log_tmp_path=$tune_output_path/matmul.log.tmp
summarize_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/summarize_scheduler.py
scheduler_log=$tune_output_path/scheduler_summary.log
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    perf_result_path=$tune_output_path/perf/${cnt}_matmul_${M}_${N}_${K}.perf
    log_result_path=$tune_output_path/log/${cnt}_matmul_${M}_${N}_${K}.log
    python ${tune_scheduler_path} -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} -s ${tune_num} -r $matmul_log_path > $perf_result_path
    cp $matmul_log_tmp_path $log_result_path
    python $summarize_scheduler_path --input $matmul_log_path --output $scheduler_log
    let cnt+=1
done

scheduler_log_output=$build_output_path/scheduler_summary.log
build_kernel_params_list_path=$PROJECT_ROOT/src/tvm_tuner/build_kernel_params_list.py
python $summarize_scheduler_path --input $scheduler_log --output $scheduler_log_output
python $build_kernel_params_list_path --scheduler_log $scheduler_log_output

touch $tune_output_path/tune.over
