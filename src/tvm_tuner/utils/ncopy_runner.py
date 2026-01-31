import traceback
from multiprocessing import Process, Barrier, Pool, Array
import multiprocessing
import time
import numpy as np
import tvm
from tvm import nd
from tvm.autotvm.measure import LocalRunner, MeasureResult, MeasureErrorNo
from tvm.error import TVMError
import os

def run_local_ncopy(
    measure_input,
    build_result,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    ref_input,
    enable_cpu_cache_flush=False,
):
    if isinstance(build_result, MeasureResult):
        return build_result

    errno = MeasureErrorNo.NO_ERROR
    try:
        # print("build_result = ")
        # print(build_result)
        lib_path = build_result.filename
        mod = tvm.runtime.load_module(lib_path)
        func = mod["default_function"]

        args_info = build_result.arg_info

        input_A_shape_info = args_info[0][0] # A or PackedA
        input_B_shape_info = args_info[1][0] # B or PackedB
        input_C_shape_info = args_info[2][0] # C

        input_A_data_type = args_info[0][1]
        input_B_data_type = args_info[1][1]
        input_C_data_type = args_info[2][1]

        dev = tvm.cpu(0)

        # def task(i):
        def task(barrier, i, costss):
            # print(f"Doing task {i} ...") # To ensure that they are 
            barrier.wait()
            os.sched_setaffinity(0, {i + 96}) # bind to i-th core
            np.random.seed(i)
            a_nd = tvm.nd.array(np.random.uniform(size=input_A_shape_info).astype(np.float32), device=dev)
            b_nd = tvm.nd.array(np.random.uniform(size=input_B_shape_info).astype(np.float32), device=dev)
            c_nd = tvm.nd.array(np.zeros(input_C_shape_info).astype(np.float32), device=dev)

            tic_task = time.time()
            for re in range(repeat):
                func(a_nd, b_nd, c_nd)
            toc_task = time.time()

            # print(f"Doing task {i} ... Done, time = {toc_task - tic_task}")
            costss[i] = toc_task - tic_task

        ncopy = 96
        costss = Array('d', range(ncopy))

        tic = time.time()

        barrier = Barrier(ncopy)
        processes = [Process(target=task, args=(barrier, i, costss)) for i in range(ncopy)]
        for p in processes:
            p.start() # 启动后都卡在barrier处
        for p in processes:
            p.join() # join后才能进行下面的操作

        toc = time.time() # 直接以ncopy完全执行完的时间作为measurement

        # print(f"All {ncopy} tasks done in time = {toc - tic}")
        # print(f"costss = {costss[:]}")

        # costs = [toc - tic]
        costs = np.average(np.array(costss[:])) / repeat

    except TVMError as exc:
        msg = str(exc)
        costs = (traceback.format_exc(), RuntimeError(msg[:1024]))
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    return MeasureResult(costs, errno, toc - tic + build_result.time_cost, tstamp)
    # return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)

class NCOPYLocalRunner(LocalRunner):
    def __init__(
        self,
        n_copy=4,
        timeout=100,
        number=4,
        repeat=5,
        **kwargs
    ):
        super(NCOPYLocalRunner, self).__init__(
            timeout=timeout,
            number=number,
            repeat=repeat,
            **kwargs
        )
        self.n_copy = n_copy

    def run(self, measure_inputs, build_results):
        results = []
        for i in range(len(measure_inputs)):
            measure_inp, build_res = measure_inputs[i], build_results[i]

            try:
                res = run_local_ncopy(
                    measure_inp,
                    build_res,
                    self.number,
                    self.repeat,
                    self.min_repeat_ms,
                    self.cooldown_interval,
                    self.ref_input,
                    self.enable_cpu_cache_flush,
                )
                results.append(res)
            except Exception as ex:
                tb = traceback.format_exc()
                res = MeasureResult(
                    (tb, ex), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time()
                )

        return results