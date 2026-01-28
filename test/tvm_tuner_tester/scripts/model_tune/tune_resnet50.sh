#!/bin/bash
set -e
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}

# parameter setting
arch=a64fx
threads=1
tune_num=10
module=resnet50

M=(64    64   64   256  64   128 128  512 512 128 256 256  1024 1024 256  512  512  2048 2048 512)
N=(12544 3136 3136 3136 3136 784 784  784 784 784 196 196  196  196  196  49   49   49   49   49)
K=(147   64   576  64   256  256 1152 128 256 512 512 2304 256  512  1024 1024 4608 512  1024 2048)

# calling tune.sh to opt
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/tune.sh
bash $TUNE_SCRIPT --arch=$arch --threads=$threads --tune_num=$tune_num --module=$module -m "${M[*]}" -n "${N[*]}" -k "${K[*]}"