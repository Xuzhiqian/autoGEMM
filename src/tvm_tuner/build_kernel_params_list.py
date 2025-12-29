import sys
import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler_log", type=str, required=True, help="scheduler_summary.log")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    print(f"Project root in build_kernel_params_list: {project_dir}")
    input_file_path = args.scheduler_log
    print(f"input_file_path in build_kernel_params_list: {input_file_path}")
    output_file_path = os.path.join(project_dir, f"src/blas_wrapper/include/kernel_params_list.hpp")
    print(f"output_file_path in build_kernel_params_list: {output_file_path}")

    if not os.path.exists(input_file_path):
        exit(-1)

    cc_code = f"""#ifndef __KERNEL_PARAMS_LIST_H_
#define __KERNEL_PARAMS_LIST_H_

#include <iostream>
#include <unordered_map>
#include <cmath>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

namespace KernelParams
{{
    struct Value {{
        int M;
        int N;
        int K;
        int64_t A_shape[2];
        int64_t B_shape[2];
        int64_t C_shape[2];

        tvm::runtime::PackedFunc func;

        uint8_t dtype_code = kDLFloat;
        uint8_t dtype_bits = 32;
        uint16_t dtype_lanes = 1;
        int device_type = kDLCPU;
        int device_id = 0;

        DLDataType dtype = {{dtype_code, dtype_bits, dtype_lanes}};
        DLDevice device = {{kDLCPU, device_id}};

        Value() : M(0), N(0), K(0) {{}}
        Value(int m, int n, int k) : M(m), N(n), K(k) {{
            A_shape[0] = M;
            A_shape[1] = K;
            B_shape[0] = K;
            B_shape[1] = N;
            C_shape[0] = M;
            C_shape[1] = N;
            std::string base_name = "GEMM_" + std::to_string(M) + "X" + std::to_string(N) + "X" + std::to_string(K);
            std::string mod_name = base_name + "_kernel.so";
            std::string func_name = "OP_" + base_name;
            std::string loading_path = "/home/linzuxuan/autoGEMM/autoGEMM/data/tune_output/build/library/" + mod_name;
            tvm::runtime::Module mod_tvmlib = tvm::runtime::Module::LoadFromFile(loading_path);

            func = mod_tvmlib.GetFunction(func_name);
        }}
    }};

    static std::map<std::string, Value> mapping;

    static void CreateMap() {{
        if (mapping.empty()) {{
"""
    with open(input_file_path, 'r') as load_f:
        for line in load_f:
            load_dict = json.loads(line)
            MNK = load_dict["input"]
            cfg = load_dict["config"]["entity"]
            M = MNK[2][0]
            N = MNK[2][1]
            K = MNK[2][2]
            cc_code+=f"""            mapping["{M}x{N}x{K}"] = {{{M}, {N}, {K}}};\n"""
    cc_code += f"""        }}
    }}
}};
#endif"""

    f = open(output_file_path, 'w')
    f.write(cc_code)
    f.close()
