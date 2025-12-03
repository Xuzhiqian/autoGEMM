CXX = /usr/bin/clang++
CFLAGS= -O3 -std=c++17 -I./include -I$(TVM_HOME)/include -I$(TVM_HOME)/3rdparty/dlpack/include -I$(TVM_HOME)/3rdparty/dmlc-core/include -L${TVM_HOME}/build -ltvm -ldl -pthread -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -Wno-macro-redefined -fPIC -shared
all:
	$(CXX) $(CFLAGS) wrapper.cpp -o lib/libautogemm_sgemm.so