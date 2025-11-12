#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

# UNROLL=4
UNROLL=8
# UNROLL=16
NR=4
TOT_REPEAT=65536000000
PIPELINE_STRATEGY_LEVEL=0
MSRA_FLAG=1

M_list=(2 3 4 5 6 7 8)
N_list=(4 8 12 16 20 24 28)
K_list=(8 16 32 64 128)

for K in ${K_list[*]}
do
    echo "K = ${K}"
	echo -n "M \ N	"
	for N in ${N_list[*]}; do
		echo -n "$N	"
	done
	echo ""
	for M in ${M_list[*]}; do
		echo -n "$M,	"
		for N in ${N_list[*]}; do
			REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
			if test $REPEAT -gt 1000000000
			then
				REPEAT=1000000000
			fi

			python ../../src/make_c_file_asm_pipeline_experiment.py $M $N $K $UNROLL $NR $REPEAT $PIPELINE_STRATEGY_LEVEL $MSRA_FLAG
			make -s
			./benchmark_kernel
		done
		echo ""
	done
done
