#ifndef _TEST_RELULAYER_IMPL
#define _TEST_RELULAYER_IMPL

#include "glog/logging.h"
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
void test_relu_forward_impl_f(
        const float* in,
        float* out,
        float negative_slope,
        int count) {
  DLOG(INFO) << " negative_slope "<< negative_slope;
  DLOG(INFO) << " count "<< count;

  int i;
  for( i = 0; i < count; ++i) {
    out[i] = max(in[i], 0.0)
              + negative_slope * min(in[i], 0.0);
  }
  printf("RELU forward-float is OK.\n");
}

void test_relu_forward_impl_d(
        const double* in,
        double* out,
        double negative_slope,
        int count) {
  DLOG(INFO) << " negative_slope "<< negative_slope;
  DLOG(INFO) << " count "<< count;
  DLOG(INFO) << " in[0] "<< in[0];
  int i;
  for( i = 0; i < count; ++i) {
    out[i] = max(in[i], 0.0)
              + negative_slope * min(in[i], 0.0);
  }
  DLOG(INFO) << " out[0] "<< out[0];
  printf("RELU forward-double is OK.\n");
}

void test_relu_backward_impl_f(
        const float* bottom_data,
        const float* top_diff,
        float* bottom_diff,
        float negative_slope,
        int count) {
  DLOG(INFO) << " negative_slope "<< negative_slope;
  DLOG(INFO) << " count "<< count;

  int i;
  for( i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
  }
  printf("RELU backward-float is OK.\n");
}

void test_relu_backward_impl_d(
        const double* bottom_data,
        const double* top_diff,
        double* bottom_diff,
        double negative_slope,
        int count) {
  DLOG(INFO) << " negative_slope "<< negative_slope;
  DLOG(INFO) << " count "<< count;
  int i;
  for( i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
  }
  printf("RELU backward-double is OK.\n");
}

#endif
