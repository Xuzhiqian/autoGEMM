#!/bin/bash
set -e
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}

# parameter setting
arch=a64fx
threads=1
tune_num=10
module=square

M=(64 128 256 512 1024 2048)
N=(64 128 256 512 1024 2048)
K=(64 128 256 512 1024 2048)

# calling tune.sh to opt
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/tune.sh
bash $TUNE_SCRIPT --arch $arch --threads $threads --tune_num $tune_num --module $module -m "${M[*]}" -n "${N[*]}" -k "${K[*]}"