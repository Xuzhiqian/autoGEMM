#!/bin/bash
set -e

# directory setting
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}
echo "Project root: $PROJECT_ROOT"
WORKING_DIR=$PROJECT_ROOT/data/tune_output
echo "Working dir: $WORKING_DIR"
time=$(date "+%Y%m%d%H%M%S")
STORING_DIR=$PROJECT_ROOT/data/scheduler_house/resnet50/$time

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

# save MNK to file for further opt
M=(64    64   64   256  64   128 128  512 512 128 256 256  1024 1024 256  512  512  2048 2048 512)
N=(12544 3136 3136 3136 3136 784 784  784 784 784 196 196  196  196  196  49   49   49   49   49)
K=(147   64   576  64   256  256 1152 128 256 512 512 2304 256  512  1024 1024 4608 512  1024 2048)

MNK_file=${WORKING_DIR}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi
LOOP_NUM=${#M[@]}
for (( i=0; i<LOOP_NUM; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> $MNK_file
done

# calling tune.sh to opt
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/tune.sh
bash $TUNE_SCRIPT $arch $threads $tune_num 

# storing tune results
if [[ -f "tune_output/tune.over" ]]; then
    echo "Storing results to $STORING_DIR"
	mkdir -p $STORING_DIR
    cp -r $WORKING_DIR/* $STORING_DIR
fi