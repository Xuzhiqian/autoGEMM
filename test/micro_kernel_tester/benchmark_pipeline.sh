#!/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
cd ${CURRENT_PATH}

# make sure lda >= K, ldb >= N, ldc >= N
# sh test_pipeline_strategy_level.sh M  N  K  lda  ldb ldc UNROLL_K NR_MAIN
sh test_pipeline_strategy_level.sh   16 1  16 3456 8   1   8        5
sh test_pipeline_strategy_level.sh   64 64 64 64   64  64  8        4