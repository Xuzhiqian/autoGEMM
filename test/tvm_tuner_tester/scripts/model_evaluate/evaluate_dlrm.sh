 #!/bin/bash
set -e
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}

# parameter setting
arch=a64fx
threads=1
repeats=1
module=dlrm

# day="20251121"
# time="20251120165501" # neon
# time="20251121161600" # sve
# day="20251126"
# time="20251126165559" # 第一次ncopy调优：100次
# time="20251126171752" # 第二次ncopy调优： 1000次
# day="20251201"
# time="20251201152719" # 比较接近kdnn性能的版本
# day="20251202"
# time="20251202113119" # 调MNK次序的版本, segmentation fault
# time="20251202214055" # 优化2000轮的版本，结果并没有current_best好
# day="20251208"
# time="20251208121337" # 所有规模的版本，性能不佳
# day="20251210"
# time="20251210160933"
# day="20251211"
# time="20251211092304"
# day="20251212"
# time="20251212092349"
# time="20251212135148" # packB和C计算图结合的版本，全都是PackedB + A x PackedB, 1000次, 81797.9
# time="20251212171859" # 3种情况结合搜索的调试版本,coredump
# time="20251212175353" # 只是性能低，没coredump


# calling evaluate.sh
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/evaluate.sh
bash $TUNE_SCRIPT --arch $arch --threads $threads --repeats $repeats --module $module --day $day --time $time
