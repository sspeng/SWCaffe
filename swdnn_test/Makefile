FLAGS=-O2 -DCPU_ONLY
BOOST_DIR=/home/export/online1/swyf/swdnn/softs/install

INC_FLAGS +=-I../../thirdparty/glog_install/include
INC_FLAGS += -I../../thirdparty/openblas_install/include
INC_FLAGS += -I../../thirdparty/googletest/include
INC_FLAGS += -I$(BOOST_DIR)
INC_FLAGS += -I../include -I./include

LDFLAGS += -L../../thirdparty/glog_install/lib/ -lglog
LDFLAGS += -L../../thirdparty/openblas_install/lib -lopenblas

THIRD_PARTY_DIR=../../thirdparty
SWINC_FLAGS=-I../include -I./include -I$(THIRD_PARTY_DIR)/CBLAS/include -I${BOOST_DIR}
SWLIBOBJ=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/libswblas0324.a


CAFFE_OBJ = 	../swdnn_test/obj/test_caffe_main.o\
							../swdnn_test/obj/test_convolution_layer.o\
							../swdnn_test/obj/test_im2col.o\
							../swdnn_test/obj/test_memcpy.o\
							../swdnn_test/obj/test_data_trans.o\
							../swdnn_test/obj/test_relu.o\
							../lib/swcaffe.a

OBJ= 		../swdnn_test/obj/test_swdnn_main.o\
							../swdnn_test/obj/test_im2col.o\
							../swdnn_test/obj/test_memcpy.o\
							../swdnn_test/obj/test_data_trans.o\
							../swdnn_test/obj/test_relu.o\
					 		../swdnn_test/obj/timer.o
SWDNN_OBJ= 	../lib/sw/swcaffe.a

#../swdnn_test/obj/conv_layer_impl.o\

all: ./bin/swdnn_main
run: ./bin/swdnn_main
	bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run ../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/swdnn_main
./bin/caffe_main: $(CAFFE_OBJ)
	mpiicc $^ ../../thirdparty/googletest/libgtest.a -o $@ $(LDFLAGS)
./bin/swdnn_main: $(SWDNN_OBJ) $(OBJ) $(SWLIBOBJ)
	cd .. && make ar
	mpiCC $^ -o $@

#intel
../swdnn_test/obj/test_caffe_main.o: ../swdnn_test/src/test_caffe_main.cpp 
	mpiicc $(FLAGS) $(INC_FLAGS) -c $^ -o $@ 
../swdnn_test/obj/test_convolution_layer.o: ../swdnn_test/src/test_convolution_layer.cpp
	mpiicc $(FLAGS) $(INC_FLAGS) -c $^ -o $@
../swdnn_test/obj/test_relu_layer.o: ../swdnn_test/src/test_relu_layer.cpp
	mpiicc $(FLAGS) $(INC_FLAGS) -c $^ -o $@

#SW26010
../swdnn_test/obj/test_im2col.o: ../swdnn_test/src/test_im2col.cpp
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@
../swdnn_test/obj/test_memcpy.o: ../swdnn_test/src/test_memcpy.cpp
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@
../swdnn_test/obj/test_data_trans.o: ../swdnn_test/src/test_data_trans.cpp
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@
../swdnn_test/obj/test_relu.o: ../swdnn_test/src/test_relu.cpp
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@
../swdnn_test/obj/timer.o: ../swdnn_test/src/timer.c
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@
../swdnn_test/obj/test_swdnn_main.o: ../swdnn_test/src/test_swdnn_main.cpp 
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@

clean:
	rm ./obj/*

mk:
	mkdir ./bin
	mkdir ../swdnn_test/obj/
