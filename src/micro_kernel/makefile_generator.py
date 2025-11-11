from global_config import *

PRECISION_MACRO = "FP64"
if PRECISION == "FLOAT":
    PRECISION_MACRO = "FP32"
if PRECISION == "FP16":
    PRECISION_MACRO = "FP16"

def makefile_generator():
    code_str = ""
    code_str += f"CXX = g++\n"
    code_str += f"CFLAGS = -march=armv8.3-a+sve -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -D{PRECISION_MACRO}\n"
    code_str += f"all:\n"
    code_str += f"\t$(CXX) $(CFLAGS) c_file_asm.cpp -o benchmark_kernel"
    return code_str
