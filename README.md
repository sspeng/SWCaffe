## Brief Introduction :
<img src="https://github.com/feifeibear/SWCaffe/blob/master/swdnnlogo.png" width = "300" height = "200" alt="swdnnlogo" align=center />
This is deep learning framework customized for chinese homemade Sunway TaihuLight supercomputer.

### Dependencies:
1. blas
http://www.netlib.org/blas/

### Features
1. No protocbuf
2. boost with only hpp headers
3. No database, such as LMDB, HDF5, etc. for data storage, read from binary file
4. Support GEMM-swBLAS and swDNN for convlayer
https://github.com/THUHPGC/swDNN.git
large image channels, swDNN is used and small channels swBLAS is used
5. support single-precison floating-point data structure for swDNN
Because the SW26010 processor dose not double the peak performance with float, we rewrote the asm code used for swDNN
6. support slave-core accelerated pooling layer
7. checkout to trans_layer, we add a trans_layer to avoid transformation 4D blobs between orignal caffe format and swDNN format
8. Suport MPI for multi-node training

### Usage
Modify Makefile to choose the hardware you want
uncomment FLAGS of swDNN in Makefile.sw to use swDNN

#### MNIST + LeNet
please download mnist data into ../data/
http://yann.lecun.com/exdb/mnist/
#### MNIST + LSTM
please download mnist data into ../data/
http://yann.lecun.com/exdb/mnist/
#### IMAGENET + ALEXNET/VGG-16
##### Please download imagenet data from https://pan.baidu.com/s/1bQdZcE password: p23u into ../data/imagenet_bin/
##### rename binary files into
    mean.bin
    test_data.bin
    test_label.bin
    test_mean.bin
    train_data.bin
    train_label.bin
    train_mean.bin
##### Prepare Caffemodel at ../data/serialized_caffemodel
###### git checkout protobuf-loadmodel
###### download VGG_ILSVRC_16_layers.caffemodel into ../data/VGG_ILSVRC_16_layers.caffemodel
###### run ./net_param_serialize

### Bugs
1. DataLayer is customized for mnist and imagenet
2. Not test for swDNN backward convlayer

###TODO
1. support multi-threading inside one CPU
2. support other data layer
3. documents for user
4. single-precison has bugs

### Developer
Jiaui Fang
Wenlai Zhao
Liandeng Li

### Contact
fang_jiarui@163.com
