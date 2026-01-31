CXX = g++
CFLAGS = -march=armv8.3-a -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -DFP32
all:
	$(CXX) $(CFLAGS) c_file_asm.cpp -o benchmark_kernel