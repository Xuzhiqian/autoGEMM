#!/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "${0}")"; pwd)
cd ${CURRENT_PATH}

# sh test_regular.sh M  N  K   UNROLL_K NR_MAIN
sh test_regular.sh   5  16 4   8        4
sh test_regular.sh   5  64 16  8        4
sh test_regular.sh   80 16 64  8        4
sh test_regular.sh   80 64 256 8        4
sh test_regular.sh   2  4  8   8        4
sh test_regular.sh   2  8  8   8        4
sh test_regular.sh   2  12 8   8        4
sh test_regular.sh   2  16 8   8        4
sh test_regular.sh   2  20 8   8        4
sh test_regular.sh   2  24 8   8        4
sh test_regular.sh   2  28 8   8        4
sh test_regular.sh   12 11 11  8        4
sh test_regular.sh   12 12 11  8        4
sh test_regular.sh   12 13 11  8        4
sh test_regular.sh   12 16 11  8        4
sh test_regular.sh   32 24 32  8        4 # zhangwei's shape
sh test_regular.sh   16 24 16  8        4
sh test_regular.sh   25 25 64  8        4
sh test_regular.sh   2  4  8   8        4
sh test_regular.sh   3  8  16  8        4
sh test_regular.sh   4  12 32  8        4
sh test_regular.sh   5  16 64  8        4
sh test_regular.sh   6  20 128  8        4
