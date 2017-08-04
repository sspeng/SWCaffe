#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif
//#include "test_conv_layer_impl.hpp"
//
using namespace std;
using namespace caffe;

typedef double Dtype;
int main() {
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* blob_bottom_2_;
  Blob<Dtype>* blob_top_;
  Blob<Dtype>* blob_top_2_;

  blob_bottom_ = new Blob<Dtype>(2, 3, 6, 4);
  blob_bottom_2_ = new Blob<Dtype>(2, 3, 6, 4);
  blob_top_ = new Blob<Dtype>();
  blob_top_2_ = new Blob<Dtype>(2,4,4,2);

  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(blob_bottom_);
  filler.Fill(blob_bottom_2_);

  blob_bottom_vec_.push_back(blob_bottom_2_);
  blob_top_vec_.push_back(blob_top_2_);

  //DLOG(INFO) << "this->blob_bottom_vec_.size() " << this->blob_bottom_vec_.size();
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  printf("caffe before setup is OK!\n");
  layer->SetUp(blob_bottom_vec_, blob_top_vec_);
  printf("caffe conv setup is OK!\n");
  layer->Forward(blob_bottom_vec_, blob_top_vec_);
  printf("caffe conv forward is OK!\n");
  // Check against reference convolution.

  return 0;
}

