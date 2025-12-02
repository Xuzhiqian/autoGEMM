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

#include <list>

namespace KernelParams
{{
    struct SimpleStruct {{
        int M;
        int N;
        int K;
        int nc;
        int kc;
        int padding_size;
        SimpleStruct(int M, int N, int K, int nc, int kc, int padding_size)
            : M(M), N(N), K(K), nc(nc), kc(kc), padding_size(padding_size){{}}
    }};
    static std::list<SimpleStruct> params_list;
    static void CreateList()
    {{
"""
    with open(input_file_path, 'r') as load_f:
        for line in load_f:
            load_dict = json.loads(line)
            MKN = load_dict["input"]
            cfg = load_dict["config"]["entity"]
            M = MKN[2][0]
            N = MKN[2][1]
            K = MKN[2][2]
            nc, kc, padding_size = 0, 0, 0
            for param_name, param_type, param_value in cfg:
                if param_name == "tile_y":
                    nc = param_value[-1]
                if param_name == "tile_k":
                    kc = param_value[-1]
                if param_name == "padding_size":
                    padding_size = param_value
            cc_code+=f"""        params_list.push_back(SimpleStruct({M}, {N}, {K}, {nc}, {kc}, {padding_size}));\n"""
    cc_code += f"""    }}
}};
#endif"""

    f = open(output_file_path, 'w')
    f.write(cc_code)
    f.close()
