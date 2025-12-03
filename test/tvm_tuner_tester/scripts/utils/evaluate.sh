#!/bin/bash
set -e

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
app="dlrm"
# day="20251121"
# time="20251120165501" # neon
# time="20251121161600" # sve
# day="20251126"
# time="20251126165559" # 第一次ncopy调优：100次
# time="20251126171752" # 第二次ncopy调优： 1000次
# day="20251201"
# time="20251201152719" # 比较接近kdnn性能的版本
day="20251202"
# time="20251202113119" # 调MNK次序的版本, segmentation fault
time="20251202214055" # 优化2000轮的版本，结果并没有current_best好
tune_output_path=$PROJECT_ROOT/data/scheduler_house/$app/$day/$time
# tune_output_path=$PROJECT_ROOT/data/scheduler_house/$app/current_best
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
python $build_kernel_params_list_path --scheduler_log $scheduler_log_output

tester_src_path=$PROJECT_ROOT/test/tvm_tuner_tester/src
cd $tester_src_path
make -s

evaluate_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/evaluate_scheduler.py
benchmark_kernel_path=$tester_src_path/benchmark_kernel
echo "benchmark_kernel_path: $benchmark_kernel_path"
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python $evaluate_scheduler_path -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} --scheduler_log ${scheduler_log_output}
    let cnt+=1
done

cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    # ldd $benchmark_kernel_path
    # LD_PRELOAD=/usr/lib64/libasan.so.8 $benchmark_kernel_path ${M} ${N} ${K} ${repeats}
    $benchmark_kernel_path ${M} ${N} ${K} ${repeats}
done

touch $build_output_path/build.over
