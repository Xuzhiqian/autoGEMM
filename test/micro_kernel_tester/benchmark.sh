#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

# UNROLL=4
UNROLL=8
# UNROLL=16
TOT_REPEAT=65536000000
PIPELINE_STRATEGY_LEVEL=0
# PIPELINE_STRATEGY_LEVEL=1
# PIPELINE_STRATEGY_LEVEL=2
# PIPELINE_STRATEGY_LEVEL=3
MSRA_FLAG=0
# MSRA_FLAG=1

# M_list=( 5  5 80 80)
# N_list=(16 64 16 64)
# K_list=( 4 16 64 256)

# M_list=(2 2 2  2  2  2  2)
# N_list=(4 8 12 16 20 24 28)
# K_list=(8 8 8  8  8  8  8)

M_list=(12 12 12 12)
N_list=(11 12 13 16)
K_list=(11 11 11 11)

# M_list=(12)
# N_list=(13)
# K_list=(11)

for K in ${K_list[*]}
do
	LOOP_NUM=${#M_list[@]}
	for (( i=0; i<$LOOP_NUM; i++))
	do
		M=${M_list[$i]} 
		N=${N_list[$i]} 

		REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
		if test $REPEAT -gt 1000000000
		then
			REPEAT=1000000000
		fi
		NR=4
		
		echo -n $M, $N, $K," " 

		python ../../src/make_c_file_asm_pipeline_experiment.py $M $N $K $UNROLL $NR $REPEAT $PIPELINE_STRATEGY_LEVEL $MSRA_FLAG
		make -s
		./benchmark_kernel

		echo ""
	done
done
