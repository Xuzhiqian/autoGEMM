#!/bin/bash
set -e

# parameter setting
arch=a64fx
threads=1
repeats=1
module=jd_mmoe
day=0
time=0

for arg in "$@"; do
    shift
    case "$arg" in
        --arch) arch="$1" ;;
        --threads) threads="$1" ;;
        --repeats) repeats="$1" ;;
        --module) module="$1" ;;
        --day) day="$1" ;;
        --time) time="$1" ;;
    esac
done

echo "-- arch is ${arch}"
echo "-- threads is ${threads}"
echo "-- repeats is ${repeats}"
echo "-- module is ${module}"
echo "-- day is ${day}"
echo "-- time is ${time}"

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
WORKING_DIR=$PROJECT_ROOT/data/tune_output
src_path=$PROJECT_ROOT/src
schedule_setting_path=$PROJECT_ROOT/data/scheduler_house/$module/$day/$time
if [[ "$day" == "0" ]] && [[ "$time" == "0" ]]; then
    schedule_setting_path=$PROJECT_ROOT/data/scheduler_house/$module/current_best
fi

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

# creating directories
if [[ -d $WORKING_DIR ]]; then
    rm -rf $WORKING_DIR
fi
mkdir -p $WORKING_DIR
mkdir -p $WORKING_DIR/build
mkdir -p $WORKING_DIR/build/gemm_obj                # 存放调优过程产生的二进制
mkdir -p $WORKING_DIR/build/gemm_source             # 存放调优过程产生的c文件
mkdir -p $WORKING_DIR/build/generated_micro_kernel  # 存放调优过程产生的micro_kernel内嵌汇编文件
mkdir -p $WORKING_DIR/build/library                 # 存放调优过程产生的tvm序列化library

MNK_file=${schedule_setting_path}/MNK.txt                       # 存放调优过程涉及的MNK规模
scheduler_log=${schedule_setting_path}/scheduler_summary.log    # 存放最终调优结果

# starting evaluation

## set environment
export OMP_NUM_THREADS=${threads}
export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

## creating kernel_params_list.hpp for wrapper and tests
build_kernel_params_list_path=$PROJECT_ROOT/src/tvm_tuner/build_kernel_params_list.py
python $build_kernel_params_list_path --scheduler_log $scheduler_log

## creating benchmark tests for autotvm generated kernel
tester_src_path=$PROJECT_ROOT/test/tvm_tuner_tester/src
cd $tester_src_path
make -s

## evaluation
evaluate_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/evaluate_scheduler.py
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python $evaluate_scheduler_path -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} --best_record_file ${scheduler_log}
done

## benchmarking
benchmark_kernel_path=$tester_src_path/benchmark_kernel
echo "benchmark_kernel_path: $benchmark_kernel_path"
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    # ldd $benchmark_kernel_path
    # LD_PRELOAD=/usr/lib64/libasan.so.8 $benchmark_kernel_path ${M} ${N} ${K} ${repeats}
    $benchmark_kernel_path ${M} ${N} ${K} ${repeats}
done

touch $WORKING_DIR/build/build.over
