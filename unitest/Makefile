CXX=mpiCC
GCC=sw5cc.new
FLAGS = 	-O2 -OPT:IEEE_arith=2
FLAGS += 	-DCPU_ONLY
FLAGS += 	-DDEBUG_VERBOSE_2
LDFLAGS = -lm_slave

THIRD_PARTY_DIR=../../thirdparty
BOOST_DIR=/home/export/online1/swyf/swdnn/softs/install
SWINC_FLAGS=-I../include -I$(THIRD_PARTY_DIR)/CBLAS/include -I${BOOST_DIR}

SWLIBOBJ=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/libswblas0324.a

#SWBUILD_DIR=../swbuild
#SRC_DIR=../src
#src=$(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/layers/*.cpp $(SRC_DIR)/solvers/*.cpp $(SRC_DIR)/util/*.cpp  $(SRC_DIR)/glog/*.cpp)
#SWOBJ=$(patsubst $(SRC_DIR)/%, $(SWBUILD_DIR)/%, $(patsubst %.cpp, %.o, $(src)))
#swdnnsrc=$(wildcard $(SRC_DIR)/swlayers/*.c $(SRC_DIR)/util/*.c)
#SWDNNOBJ=$(patsubst $(SRC_DIR)/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swdnnsrc)))
#SWDNNOBJ+=$(SWBUILD_DIR)/swlayers/gemm_asm.o
#SWDNNOBJ+=$(SWBUILD_DIR)/swlayers/gemm_asm_float.o
#SWOBJ+=$(SWDNNOBJ)


runconv: test_conv_layer 
	bsub -b -I -m 1 -p -q q_sw_expr -host_stack 1000 -share_size 5000 -n 1 -cgsp 64 ./$^
test_conv_layer: ./test_conv_layer.o ../lib/sw/swcaffe.a $(SWLIBOBJ)
	$(CXX) $^ $(LDFLAGS) -o $@
#conv: ./test_conv_layer.o $(SWOBJ) $(SWLIBOBJ)
./test_conv_layer.o: ./test_conv_layer.cpp
	$(CXX) -c $^ -host $(FLAGS) $(SWINC_FLAGS) -o $@
clean:
	rm *.o test_conv_layer

