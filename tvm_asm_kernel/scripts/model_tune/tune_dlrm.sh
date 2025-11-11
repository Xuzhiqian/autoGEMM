#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../..; pwd`
cd ${PROJECT_ROOT}

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

M=(128  128  128 128 128  128)
N=(3456 512  13  256 3456 512)
K=(512  3456 512 128 1    256)

# M=(128 )
# N=(3456)
# K=(512 )

# M=(128)
# N=(512)
# K=(3456)

MNK_file=${PROJECT_ROOT}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi

LOOP_NUM=${#M[@]}
for (( i=0; i<LOOP_NUM; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> MNK.txt
done

bash ./scripts/utils/tune.sh $arch $threads $tune_num 

time=$(date "+%Y%m%d%H%M%S")
if [[ -f "tune_output/tune.over" ]]; then
	mkdir -p scheduler_house/dlrm/$time
    cp -r tune_output/* scheduler_house/dlrm/$time/
    cp -r build/* scheduler_house/dlrm/$time/
fi
