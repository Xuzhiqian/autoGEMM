#!/bin/bash
set -e

# parameter setting
arch=a64fx
threads=1
tune_num=10
module=jd_mmoe
M=()
N=()
K=()

for arg in "$@"; do
    shift
    case "$arg" in
        --arch) arch="$1" ;;
        --threads) threads="$1" ;;
        --tune_num) tune_num="$1" ;;
        --module) module="$1" ;;
        -m) read -ra M <<< "$1";;
        -n) read -ra N <<< "$1";;
        -k) read -ra K <<< "$1";;
    esac
done

echo "-- arch is ${arch}"
echo "-- threads is ${threads}"
echo "-- tune_num is ${tune_num}"
echo "-- module is ${module}"
echo "-- M is ${M[@]}"
echo "-- N is ${N[@]}"
echo "-- K is ${K[@]}"

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
WORKING_DIR=$PROJECT_ROOT/data/tune_output
src_path=$PROJECT_ROOT/src
day=$(date "+%Y%m%d")
time=$(date "+%Y%m%d%H%M%S")
STORING_DIR=$PROJECT_ROOT/data/scheduler_house/$module/$day/$time

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

# creating directories
if [[ -d "$WORKING_DIR" ]]; then
    rm -rf $WORKING_DIR
fi
mkdir -p $WORKING_DIR
mkdir -p $WORKING_DIR/build
mkdir -p $WORKING_DIR/build/gemm_obj                # 存放调优过程产生的二进制
mkdir -p $WORKING_DIR/build/gemm_source             # 存放调优过程产生的c文件
mkdir -p $WORKING_DIR/build/generated_micro_kernel  # 存放调优过程产生的micro_kernel内嵌汇编文件
mkdir -p $WORKING_DIR/build/library                 # 存放调优过程产生的tvm序列化library
mkdir -p $WORKING_DIR/perf                          # 存放调优过程产生的perf中间过程
mkdir -p $WORKING_DIR/log                           # 存放调优过程产生的中间参数及结果

MNK_file=$WORKING_DIR/MNK.txt                       # 存放调优过程涉及的MNK规模
matmul_log_path=$WORKING_DIR/matmul.log             # 调优过程的所有参数及结果
scheduler_log=$WORKING_DIR/scheduler_summary.log    # 存放最终调优结果
touch $MNK_file
touch $matmul_log_path
touch $scheduler_log

# starting tuning

## create MNK.txt
MNK_file=${WORKING_DIR}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi
LOOP_NUM=${#M[@]}
echo $LOOP_NUM
for (( i=0; i<LOOP_NUM; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> $MNK_file
done

## set environment
export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

cnt=0
tune_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/tune_scheduler.py
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    perf_result_path=$WORKING_DIR/perf/${cnt}_matmul_${M}_${N}_${K}.perf
    log_result_path=$WORKING_DIR/log/${cnt}_matmul_${M}_${N}_${K}.log
    python ${tune_scheduler_path} -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} -s ${tune_num} --record_file $matmul_log_path --best_record_file $scheduler_log | tee $perf_result_path
    cp $matmul_log_path $log_result_path
    let cnt+=1
done

build_kernel_params_list_path=$PROJECT_ROOT/src/tvm_tuner/build_kernel_params_list.py
python $build_kernel_params_list_path --scheduler_log $scheduler_log

# storing tune results
touch $WORKING_DIR/tune.over
if [[ -f "$WORKING_DIR/tune.over" ]]; then
    echo "Storing results to $STORING_DIR"
	mkdir -p $STORING_DIR
    cp -r $WORKING_DIR/* $STORING_DIR
fi