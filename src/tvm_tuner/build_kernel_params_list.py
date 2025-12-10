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

namespace KernelParams
{{
    struct Value {{
        int M;
        int N;
        int K;
        int bm;
        int bn;
        int bk;
        int padding_size;
        int bn_ceil;
        int bk_ceil;
        int packedA_size;
        int packedB_size;
        int packAB;
        int64_t A_shape[2];
        int64_t packedA_shape[4];
        int64_t B_shape[2];
        int64_t packedB_shape[4];
        int64_t C_shape[2];

        tvm::runtime::PackedFunc pack_func;
        tvm::runtime::PackedFunc func;

        Value() : M(0), N(0), K(0), bm(0), bn(0), bk(0), padding_size(0), packAB(0) {{}}
        Value(int m, int n, int k, int bm, int bn, int bk, int p, int packAB) : M(m), N(n), K(k), bn(bn), bk(bk), padding_size(p), packAB(packAB) {{
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

            if (packAB == 0) {{
                ;
            }} else if (packAB == 1) {{
                bk_ceil = ((bk - 1) / padding_size + 1) * padding_size;
                packedA_size = M * (K / bk) * bk_ceil;
                packedA_shape[0] = M / bm;
                packedA_shape[1] = K / bk;
                packedA_shape[2] = bm;
                packedA_shape[3] = bk_ceil;

                std::string pack_func_name = func_name + "_packA";
                pack_func = mod_tvmlib.GetFunction(pack_func_name);
            }} else if (packAB == 2) {{
                bn_ceil = ((bn - 1) / padding_size + 1) * padding_size;
                packedB_size = K * (N / bn) * bn_ceil;
                packedB_shape[0] = K / bk;
                packedB_shape[1] = N / bn;
                packedB_shape[2] = bk;
                packedB_shape[3] = bn_ceil;

                std::string pack_func_name = func_name + "_packB";
                pack_func = mod_tvmlib.GetFunction(pack_func_name);
            }}
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
            bm, bn, bk, padding_size, packAB = 0, 0, 0, 0, 0
            for param_name, param_type, param_value in cfg:
                if param_name == "tile_x":
                    bm = param_value[-1]
                if param_name == "tile_y":
                    bn = param_value[-1]
                if param_name == "tile_k":
                    bk = param_value[-1]
                if param_name == "padding_size":
                    padding_size = param_value
                if param_name == "packAB":
                    packAB = param_value
            cc_code+=f"""            mapping["{M}x{N}x{K}"] = {{{M}, {N}, {K}, {bm}, {bn}, {bk}, {padding_size}, {packAB}}};\n"""
    cc_code += f"""        }}
    }}
}};
#endif"""

    f = open(output_file_path, 'w')
    f.write(cc_code)
    f.close()
