#!/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
cd ${CURRENT_PATH}

M=$1
N=$2
K=$3
UNROLL_K=$4
NR_MAIN=$5
REPEAT=64

for PIPELINE_STRATEGY_LEVEL in 0 1 2 3; do
	for MRSA_FLAG in 0 1; do
		# "regular": assuming lda=K, ldb=M, ldc=N
		python ${CURRENT_PATH}/test.py --M $M --N $N --K $K --lda $K --ldb $M --ldc $N --UNROLL_K $UNROLL_K --NR_MAIN $NR_MAIN --REPEAT $REPEAT --PIPELINE_STRATEGY_LEVEL $PIPELINE_STRATEGY_LEVEL --MRSA_FLAG $MRSA_FLAG
	done
done
