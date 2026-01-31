#!/bin/bash
set -e
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}

# parameter setting
arch=a64fx
threads=1
tune_num=20
module=jd_mmoe

# M=(128 128 128 128 128 128 128  128 )
# N=(1   2   2   32  64  128 128  256 )
# K=(2   32  64  64  128 256 1440 1440)

M=(128  128 )
N=(256  128 )
K=(1440 1440)

# calling tune.sh to opt
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/tune.sh
bash $TUNE_SCRIPT --arch $arch --threads $threads --tune_num $tune_num --module $module -m "${M[*]}" -n "${N[*]}" -k "${K[*]}"