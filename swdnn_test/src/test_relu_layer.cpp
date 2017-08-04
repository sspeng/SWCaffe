#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/relu_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
//#include "caffe/swlayers/conv_layer_impl.hpp"
#include "test_relu_layer_impl.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_relu_forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template void caffe_relu_forward(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_relu_forward(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename Dtype>
void caffe_relu_backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}

template void caffe_relu_backward(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_relu_backward(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ReluLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReluLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ReluLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReluLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReluLayerTest, TestSimpleRelu) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  DLOG(INFO) << "this->blob_bottom_vec_.size() " << this->blob_bottom_vec_.size();
  LayerParameter layer_param;
  ReLUParameter* relu_param =
      layer_param.mutable_relu_param();
  // yx
  //convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ReLULayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  //YX subtitude with my conv impl
  caffe_relu_forward(this->blob_bottom_,
      this->MakeReferenceTop(this->blob_top_));
/*
  test_conv_forward_impl<Dtype>(this->blob_bottom_->mutable_cpu_data(), 
        layer->blobs()[0]->mutable_cpu_data(), 
        this->MakeReferenceTop(this->blob_top_)->mutable_cpu_data(),
        layer->blobs()[1]->mutable_cpu_data(), 
        //int Ci,
        this->blob_bottom_->width(),
        //int Ri,
        this->blob_bottom_->height(),
        //int K,
        3,
        //int Ni,
        this->blob_bottom_->channels(),
        //int No,
        this->blob_top_->channels(),
        //int B,
        this->blob_top_->num());*/
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_relu_forward(this->blob_bottom_2_,
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ReluLayerTest, TestSWBackward) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 5;
  const int kernel_w = 5;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 100;
  bottom_shape[1] = 20;
  bottom_shape[2] = 5; //kernel_h * 2;
  bottom_shape[3] = 5; //kernel_w * 2;
  DLOG(INFO) << " this->blob_bottom_vec_.size() " << this->blob_bottom_vec_.size();
  DLOG(INFO) << " this->blob_bottom_vec_[0]->shape " << this->blob_bottom_vec_[0]->shape_string();
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
  }
  LayerParameter layer_param;
  ReLUParameter* relu_param =
      layer_param.mutable_relu_param();

  Blob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ReLULayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    ASSERT_EQ(2, layer.blobs().size());
    copy_diff = false; reshape = true;
    //bias.CopyFrom(*layer.blobs()[1], copy_diff, reshape);
  }
  vector<bool> propagate_down(1, true);
  Blob<Dtype> result_2d;
  Blob<Dtype> backward_result_2d;
  // Test with 2D im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    //caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    //caffe_set(bias.count(), Dtype(0), bias.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result.
    //relu_param->set_force_nd_im2col(false);
    ReLULayer<Dtype> layer_2d(layer_param);
    layer_2d.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(2, layer_2d.blobs().size());
    copy_diff = false; reshape = false;
    //layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    //layer_2d.blobs()[1]->CopyFrom(bias, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    //Backward
    layer_2d.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);

    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    //backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
    //backward_weight_result_2d.CopyFrom(*layer_2d.blobs()[0], 
    //    copy_diff, reshape);
    //backward_bias_result_2d.CopyFrom(*layer_2d.blobs()[1],
    //    copy_diff, reshape);

    //my code
    copy_diff = true; reshape = true;
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    DLOG(INFO) << "begin my backward";
    DLOG(INFO) << " this->blob_bottom_vec_[0] shape " << this->blob_bottom_vec_[0]->shape_string();
    DLOG(INFO) << " this->blob_top_vec_[0] shape " << this->blob_top_vec_[0]->shape_string();
    DLOG(INFO) << " layer_2d.blobs() shape " << layer_2d.blobs()[0]->shape_string();
    DLOG(INFO) << " this->blob_bottom_vec_[0]->width() " <<
        this->blob_bottom_vec_[0]->width();
    DLOG(INFO) << " this->blob_bottom_vec_[0]->height() " << this->blob_bottom_vec_[0]->height();

    DLOG(INFO) << " this->blob_bottom_vec_[0]->channels() " <<
        this->blob_bottom_vec_[0]->channels();
    DLOG(INFO) << " this->blob_top_vec_[0]->num() " <<
        this->blob_top_vec_[0]->num();

    layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_2d.blobs()[1]->CopyFrom(bias, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
//#define ORI
#ifdef ORI
    layer_2d.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
#else
    caffe_relu_backward(this->blob_top_vec_,propagate_down,this->blob_bottom_vec_);
    /*
    test_conv_backward_impl<Dtype>(this->blob_bottom_vec_[0]->cpu_data(),
        //Type* out_grad,
        this->blob_top_vec_[0]->cpu_diff(),
        //Type* weight,
        layer_2d.blobs()[0]->cpu_data(),
        //Type* in_grad,
        this->blob_bottom_vec_[0]->mutable_cpu_diff(),
        //Type* weight_diff,
        layer_2d.blobs()[0]->mutable_cpu_diff(),
        //Type* bias_grad,
        layer_2d.blobs()[1]->mutable_cpu_diff(),
        //int Ci,
        this->blob_bottom_vec_[0]->width(),
        //int Ri,
        this->blob_bottom_vec_[0]->height(),
        //int K,
        //kernel_w,
        layer_2d.kernel_shape().cpu_data()[0],
        //int Ni,
        this->blob_bottom_vec_[0]->channels(),
        //int No,
        this->blob_top_vec_[0]->channels(),
        //int B);
        this->blob_top_vec_[0]->num());*/
#endif

        ASSERT_EQ(backward_result_2d.count(), this->blob_bottom_vec_[0]->count());
        for (int i = 0; i < backward_result_2d.count(); ++i) {
          EXPECT_NEAR(backward_result_2d.cpu_diff()[i],
                    this->blob_bottom_vec_[0]->cpu_diff()[i], 1e-4);
        }
        //ASSERT_EQ(layer_2d.blobs()[1]->count(), backward_bias_result_2d.count());
        //for (int i = 0; i < backward_bias_result_2d.count(); ++i) {
        //  EXPECT_NEAR(backward_bias_result_2d.cpu_diff()[i],
        //            layer_2d.blobs()[1]->cpu_diff()[i], 1e-4);
        //}
        //ASSERT_EQ(backward_weight_result_2d.count(),
        //          layer_2d.blobs()[0]->count());
        //for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
        //  EXPECT_NEAR(backward_weight_result_2d.cpu_diff()[i],
        //            layer_2d.blobs()[0]->cpu_diff()[i], 1e-4);
        //}
  }
  DLOG(INFO) << "backward OK";

}

}  // namespace caffe
