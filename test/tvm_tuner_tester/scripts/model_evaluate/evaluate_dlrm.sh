 #!/bin/bash
set -e

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
WORKING_DIR=$PROJECT_ROOT/data/tune_output
echo "Working dir: $WORKING_DIR"
app="dlrm"
day="20251112"
time="20251112101735"
STORING_DIR=$PROJECT_ROOT/data/scheduler_house/$app/$day/$time
src_path=$PROJECT_ROOT/src

# parameter setting
arch=$1
threads=$2

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

export PYTHONPATH=$PYTHONPATH:$src_path:$src_path/tvm_tuner:$src_path/micro_kernel
export TVM_CC=clang++

evaluate_scheduler_path=$PROJECT_ROOT/src/tvm_tuner/evaluate_scheduler.py
best_scheduler_log_path=$STORING_DIR/scheduler_summary.log
MNK_file=${STORING_DIR}/MNK.txt 
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python $evaluate_scheduler_path -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} --scheduler_log $best_scheduler_log_path
    let cnt+=1
done
