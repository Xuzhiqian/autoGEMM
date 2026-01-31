 #!/bin/bash
set -e
tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../../../..; pwd`
cd ${PROJECT_ROOT}

# parameter setting
arch=a64fx
threads=1
repeats=1
module="jd_mmoe"

# day="20260122"
# time="20260122222523"
# day="20260123"
# time="20260123142351"
# day="20260127"
# time="20260127165320"
day="20260128"
time="20260128104903"

# calling evaluate.sh
TUNE_SCRIPT=$PROJECT_ROOT/test/tvm_tuner_tester/scripts/utils/evaluate.sh
bash $TUNE_SCRIPT --arch $arch --threads $threads --repeats $repeats --module $module --day $day --time $time
