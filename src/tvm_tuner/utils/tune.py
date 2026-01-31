from global_config import *
import tvm
from tvm import te
from tvm import autotvm

import os
from config.common_config import measure_option

# logging config (for printing tuning log to the screen)
import logging
import sys
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def tune(
    M,
    N,
    K,
    record_file,
    best_record_file,
    parallel, 
    n_trial=2500,
    early_stopping=1000,
    target="llvm",
):
    task = autotvm.task.create(
        "matmul", args=[M, N, K, parallel], target=target
    )
    logger.info(task.config_space)

    # tuner = autotvm.tuner.XGBTuner(task)
    tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
    # tuner = autotvm.tuner.GridSearchTuner(task)

    tuner.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(n_trial, prefix='\n'),
            autotvm.callback.log_to_file(record_file),
        ],
    )