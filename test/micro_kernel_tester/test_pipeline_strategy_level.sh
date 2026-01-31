#!/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
cd ${CURRENT_PATH}

M=$1
N=$2
K=$3
lda=$4
ldb=$5
ldc=$6
UNROLL_K=$7
NR_MAIN=$8

TOT_REPEAT=65536000000
REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
if test $REPEAT -gt 1000000000
then
	REPEAT=1000000000
fi

for PIPELINE_STRATEGY_LEVEL in 0 1 2 3; do
	for MSRA_FLAG in 0 1; do
		sh ${CURRENT_PATH}/test.sh --M $M --N $N --K $K --lda $lda --ldb $ldb --ldc $ldc --UNROLL_K $UNROLL_K --NR_MAIN $NR_MAIN --REPEAT $REPEAT --PIPELINE_STRATEGY_LEVEL $PIPELINE_STRATEGY_LEVEL --MSRA_FLAG $MSRA_FLAG
	done
done
