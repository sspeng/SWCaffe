extern "C" {
#include "caffe/swlayers/sw_conv_layer_impl.h"
#include "caffe/swlayers/sw_relu_layer_impl.h"
#include "caffe/util/data_type_trans.h"
#include "athread.h"
}
#include "caffe/swlayers/conv_layer_impl.hpp"
#include "test_relu_layer_impl.hpp"
#include <math.h>
//#include <iostream>
//using namespace std;
#include <stdio.h>
#include "athread.h"
#include "timer.h"

void test_forward_pad_float() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 1;
  pad = 3;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out = (float*)malloc(sizeof(float)*out_size);
  float* out_ref_ori = (float*)malloc(sizeof(float)*out_size);

  double* in_d = (double*)malloc(sizeof(double)*in_size);
  double* weight_d = (double*)malloc(sizeof(double)*weight_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < in_size; ++i )
    in_d[i] = in[i];

  for( int i = 0; i < weight_size; ++i )
    weight_d[i] = weight[i];

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
    out_ref_ori[i] = 0;
  }


  for( int st = 0; st < 1; ++st ){
    printf("running sw version pad conv...\n");
    begin_timer("sw_conv_pad_forward_impl_f");
    sw_conv_forward_pad_impl_f(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    stop_timer();
    printf("sw version pad conv OK\n");

    begin_timer("sw_conv_pad_forward_impl_d");
    sw_conv_forward_pad_impl_d(
        in_d,
        weight_d,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    stop_timer();

    begin_timer("sw_conv_pad_forward_impl_f_ori");
    sw_conv_forward_pad_impl_f_ori(
        in,
        weight,
        out_ref_ori,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    stop_timer();

    printf("inner loop OK!\n");
  }

  printf("calculating errors...\n");
  float sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 ) {
     printf("ERROR at %d: %f vs %f\n", i, out_ref[i], out[i]);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
   }
   sum += out[i];
   sum_ref += out_ref[i];
  }
  if( fabs(sum_ref - sum) > 1e-4 ) {
     printf("ERROR at SUM: %f vs %f\n", sum_ref, sum);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
  }
  printf("sum %f vs sum_ref %f athread forward OK!\n", sum, sum_ref);


  free(out_ref);
  free(out);
  free(in);
  free(weight);

}

void test_forward_pad_float_fast() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 1;
  pad = 3;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  double* in_d = (double*)malloc(sizeof(double)*in_size);
  double* weight_d = (double*)malloc(sizeof(double)*weight_size);
  float* out = (float*)malloc(sizeof(float)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i ) {
    in[i] = rand()/(float)RAND_MAX;
    in_d[i] = (double)in[i];
  }

  for( int i = 0; i < weight_size; ++i ) {
    weight[i] = rand()/(float)RAND_MAX;
    weight_d[i] = (double)weight[i];
  }

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
  }


    begin_timer("sw_conv_pad_forward_impl_f_fast");
    sw_conv_forward_pad_impl_f_fast(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    stop_timer();
    printf("sw version pad conv OK\n");
#ifdef CHECKRES
    conv_forward_pad_impl<double>(
        in_d,
        weight_d,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  float sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%f vs %f\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  printf("sum %f vs sum_ref %f athread forward OK!\n", sum, sum_ref);
#endif

  free(out_ref);
  free(out);
  free(in);
  free(weight);
  free(in_d);
  free(weight_d);

}

void test_forward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 1;
  pad = 3;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out = (double*)malloc(sizeof(double)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
  }


  for( int st = 0; st < 1; ++st ){
    sw_conv_forward_pad_impl_d(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv OK\n");

    conv_forward_pad_impl<double>(
        in,
        weight,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");

  double sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%lf vs %lf\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

  free(out_ref);
  free(out);
  free(in);
  free(weight);

}

void test_forward() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  Ni = 128;
  No = 256;
  B  = 128;
  Co = 2;
  Ro = 2;
  K  = 3;
  Ci = Co + K - 1;
  Ri = Ro + K - 1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out = (double*)malloc(sizeof(double)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;


  for( int st = 0; st < 1; ++st ){
    sw_conv_forward_impl_d(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

    conv_forward_impl<double>(
        in,
        weight,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);
    printf("inner loop OK!\n");
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");

  double sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%lf vs %lf\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  free(out_ref);
  free(out);
  free(in);
  free(weight);
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

}

int test_backward() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  Ni = 128;
  No = 128;
  B  = 128;
  Co = 2;
  Ro = 2;
  K  = 3;
  Ci = Co + K - 1;
  Ri = Ro + K - 1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  sw_conv_backward_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

  conv_backward_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);

  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);


}

int test_backward_pad_float() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* in_diff = (float*)malloc(sizeof(float)*in_size);
  float* in_diff_ref = (float*)malloc(sizeof(float)*in_size);
  float* weight_diff = (float*)malloc(sizeof(float)*weight_size);
  float* weight_diff_ref = (float*)malloc(sizeof(float)*weight_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out_diff = (float*)malloc(sizeof(float)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(float)RAND_MAX;
  begin_timer("sw_conv_pad_backward_impl_f");
  sw_conv_backward_pad_impl_f(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  stop_timer();
  conv_backward_pad_impl<float>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-2)
      printf("in_diff %f vs ref %f\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-2)
      printf("weight_diff %f vs ref %f\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad_float_fast() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* in_diff = (float*)malloc(sizeof(float)*in_size);
  float* in_diff_ref = (float*)malloc(sizeof(float)*in_size);
  float* weight_diff = (float*)malloc(sizeof(float)*weight_size);
  float* weight_diff_ref = (float*)malloc(sizeof(float)*weight_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out_diff = (float*)malloc(sizeof(float)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(float)RAND_MAX;
  begin_timer("sw_conv_pad_backward_impl_f_fast");
  sw_conv_backward_pad_impl_f_fast(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  stop_timer();
  conv_backward_pad_impl<float>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-2)
      printf("in_diff %f vs ref %f\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-2)
      printf("weight_diff %f vs ref %f\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad_split_float_fast() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* in_diff = (float*)malloc(sizeof(float)*in_size);
  float* in_diff_ref = (float*)malloc(sizeof(float)*in_size);
  float* weight_diff = (float*)malloc(sizeof(float)*weight_size);
  float* weight_diff_ref = (float*)malloc(sizeof(float)*weight_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out_diff = (float*)malloc(sizeof(float)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(float)RAND_MAX;
  begin_timer("sw_conv_pad_backward_impl_f_fast");
  sw_conv_backward_pad_weight_diff_impl_f_fast(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  sw_conv_backward_pad_in_diff_impl_f_fast(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  stop_timer();
  conv_backward_pad_impl<float>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-2)
      printf("in_diff %f vs ref %f\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-2)
      printf("weight_diff %f vs ref %f\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  begin_timer("sw_conv_pad_backward_impl_d");
  sw_conv_backward_pad_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  stop_timer();
  conv_backward_pad_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");
  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad_split() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  begin_timer("sw_conv_pad_backward_impl_d");
  sw_conv_backward_pad_weight_diff_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  sw_conv_backward_pad_in_diff_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  stop_timer();

#ifdef CHECKRES
  conv_backward_pad_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
#endif
  printf("backward test OK!");
  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

void test_relu_forward_f() {
  printf("Test relu forward-float...\n");
  int count;
  float negative_slope;
  count = 2*1024*1024; // 16M
  negative_slope = rand()/(float)RAND_MAX - 0.5;

  float* in = (float*)malloc(sizeof(float)*count);
  float* out = (float*)malloc(sizeof(float)*count);
  float* out_ref = (float*)malloc(sizeof(float)*count);

  for( int i = 0; i < count; ++i )
    in[i] = rand()/(float)RAND_MAX - 0.5;

  for( int st = 0; st < 1; ++st ){
    begin_timer("sw_relu_forward_impl_f");
    sw_relu_forward_impl_f(
        in,
        out,
        negative_slope,
        count);
    stop_timer();
    begin_timer("test_relu_forward_impl_f");
    test_relu_forward_impl_f(
        in,
        out_ref,
        negative_slope,
        count);
    stop_timer();
    printf("inner loop %d OK!\n",st);
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");
  printf("now calculating errors.\n");
  double sum = 0, sum_ref = 0;
  for( int i = 0; i < count; ++i ) {
  #ifdef DEBUG_INFO
    printf("%d: input: %lf\n",i,in[i]);
    printf("%d: swdnn-relu:%lf ori-relu:%lf\n",i,out[i],out_ref[i]);
    printf("%d: swdnn-relu - ori-relu = %lf\n",i,out[i]-out_ref[i]);
  #endif
   if( fabs(out_ref[i] - out[i]) > 1e-4) {
     printf("ERROR at %d: %lf vs %lf\n", i, out_ref[i], out[i]);
     printf("\tinput: %lf negative_slope: %lf\n", in[i],negative_slope);
  printf("\tlocal_count for id=0: %d\n",count/64 + (0<(count%64)));
  printf("\tlocal_start for id=1: %d\n",(count/64)+(1<(count%64)));
#define DEBUG
#ifdef DEBUG
  void *param_in = in;
  float* in_ptr = &((float*)param_in)[(count/64)+(1<(count%64))];
  printf("\t%lf\n",in_ptr[0]);
  printf("\tsearching...\n");
  for(int j = 0; j<count; ++j ) {
    if(fabs(out_ref[j] - out[i]) < 1e-4)
      printf("Maybe this one: index=%d, out_ref=%lf, out=%lf\n",j,out_ref[j],out[i]);
      break;
  }
#endif
     printf("************** athread forward failed ****************\n");
     free(out_ref);
     free(out);
     free(in);
     return ;
   }
   sum += out[i];
   sum_ref += out_ref[i];
  }
  free(out_ref);
  free(out);
  free(in);
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

}

void test_relu_backward_f() {
  printf("Test relu backward-float...\n");
  int count;
  float negative_slope;
  count = 2*1024*1024;
  negative_slope = rand()/(float)RAND_MAX - 0.5;
  int in_size     = count;
  int out_size    = count;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* in_diff = (float*)malloc(sizeof(float)*in_size);
  float* in_diff_ref = (float*)malloc(sizeof(float)*in_size);
  float* out_diff = (float*)malloc(sizeof(float)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX - 0.5;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(float)RAND_MAX - 0.5;

  printf("now calculating swdnn value...\n");
    begin_timer("sw_relu_backward_impl_f");
    sw_relu_backward_impl_f(
        in,
        out_diff,
        in_diff,
        negative_slope,
        count);
    stop_timer();
  printf("now calculating reference value...\n");
    begin_timer("test_relu_backward_impl_f");
    test_relu_backward_impl_f(
        in,
        out_diff,
        in_diff_ref,
        negative_slope,
        count);
    stop_timer();

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4) {
      printf("ERROR at %d: in_diff %lf vs ref %lf\n", i, in_diff[i], in_diff_ref[i]);
      printf("\tinput: %lf out_diff: %lf negative_slope: %lf\n",in[i],out_diff[i],negative_slope);
      printf("************** athread backward failed ****************\n");
      free(in_diff_ref);
      free(out_diff);
      free(in);
      free(in_diff);
      return ;
    }

  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(out_diff);


}

void test_relu_forward() {
  printf("Test relu forward...\n");
  int count;
  double negative_slope;
  count = 2*1024*1024; // 16M
  negative_slope = rand()/(double)RAND_MAX - 0.5;

  double* in = (double*)malloc(sizeof(double)*count);
  double* out = (double*)malloc(sizeof(double)*count);
  double* out_ref = (double*)malloc(sizeof(double)*count);

  for( int i = 0; i < count; ++i )
    in[i] = rand()/(double)RAND_MAX - 0.5;

  for( int st = 0; st < 1; ++st ){
    begin_timer("sw_relu_forward_impl_d");
    sw_relu_forward_impl_d(
        in,
        out,
        negative_slope,
        count);
    stop_timer();
    begin_timer("test_relu_forward_impl_d");
    test_relu_forward_impl_d(
        in,
        out_ref,
        negative_slope,
        count);
    stop_timer();
    printf("inner loop %d OK!\n",st);
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");
  printf("now calculating errors.\n");
  double sum = 0, sum_ref = 0;
  for( int i = 0; i < count; ++i ) {
  #ifdef DEBUG_INFO
    printf("%d: input: %lf\n",i,in[i]);
    printf("%d: swdnn-relu:%lf ori-relu:%lf\n",i,out[i],out_ref[i]);
    printf("%d: swdnn-relu - ori-relu = %lf\n",i,out[i]-out_ref[i]);
  #endif
   if( fabs(out_ref[i] - out[i]) > 1e-4) {
     printf("ERROR at %d: %lf vs %lf\n", i, out_ref[i], out[i]);
     printf("\tinput: %lf negative_slope: %lf\n", in[i],negative_slope);
  printf("\tlocal_count for id=0: %d\n",count/64 + (0<(count%64)));
  printf("\tlocal_start for id=1: %d\n",(count/64)+(1<(count%64)));
#define DEBUG
#ifdef DEBUG
  void *param_in = in;
  double* in_ptr = &((double*)param_in)[(count/64)+(1<(count%64))];
  printf("\t%lf\n",in_ptr[0]);
  printf("\tsearching...\n");
  for(int j = 0; j<count; ++j ) {
    if(fabs(out_ref[j] - out[i]) < 1e-4)
      printf("Maybe this one: index=%d, out_ref=%lf, out=%lf\n",j,out_ref[j],out[i]);
      break;
  }
#endif
     printf("************** athread forward failed ****************\n");
     free(out_ref);
     free(out);
     free(in);
     return ;
   }
   sum += out[i];
   sum_ref += out_ref[i];
  }
  free(out_ref);
  free(out);
  free(in);
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

}

void test_relu_backward() {
  printf("Test relu backward...\n");
  int count;
  double negative_slope;
  count = 2*1024*1024;
  negative_slope = rand()/(double)RAND_MAX - 0.5;
  int in_size     = count;
  int out_size    = count;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX - 0.5;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX - 0.5;

  printf("now calculating swdnn value...\n");
    begin_timer("sw_relu_backward_impl_d");
    sw_relu_backward_impl_d(
        in,
        out_diff,
        in_diff,
        negative_slope,
        count);
    stop_timer();
  printf("now calculating reference value...\n");
    begin_timer("test_relu_backward_impl_d");
    test_relu_backward_impl_d(
        in,
        out_diff,
        in_diff_ref,
        negative_slope,
        count);
    stop_timer();

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4) {
      printf("ERROR at %d: in_diff %lf vs ref %lf\n", i, in_diff[i], in_diff_ref[i]);
      printf("\tinput: %lf out_diff: %lf negative_slope: %lf\n",in[i],out_diff[i],negative_slope);
      printf("************** athread backward failed ****************\n");
      free(in_diff_ref);
      free(out_diff);
      free(in);
      free(in_diff);
      return ;
    }

  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(out_diff);


}

void test_double2float(int count,const char* n) {
#define SRC_TYPE double
#define DST_TYPE float
  printf("Test double to float conversion...\n");
  //int count = 32*1024*1024; // 8*32M double data;
  SRC_TYPE* src = (SRC_TYPE*)malloc(count*sizeof(SRC_TYPE));
  DST_TYPE* dst = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  DST_TYPE* dst_ref = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  printf("creating data... SIZE:%d\n",count);
  for( int i = 0; i < count; ++i )
    src[i] = rand()/(SRC_TYPE)RAND_MAX - 0.5;

  printf("calling sw_double2float.\n");
  char str1[100] = "Accelerated double2float ";
  strcat(str1,n);
  begin_timer(str1);
  double2float(src,dst,count);
  stop_timer();

  printf("calculating reference value.\n");
  char str2[100] = "Reference double2float ";
  strcat(str2,n);
  begin_timer(str2);
  for( int i = 0; i < count; ++i )
    dst_ref[i] = (DST_TYPE)src[i];
  stop_timer();

  printf("calculating errors...\n");
  float sum=0.0, sum_ref=0.0;
  printf("inout at 0: In:%lf Ref:%f vs SW:%f\n",src[0],dst_ref[0],dst[0]);
  for( int i = 0; i < count; ++i ) {
    sum+=dst[i];
    sum_ref+=dst_ref[i];
    if(fabs(dst_ref[i] - dst[i])>1e-4) {
      printf("WRONG at %d: Ref:%lf vs SW:%lf\n",i,dst_ref[i],dst[i]);
      printf("\tIn:%lf\n",src[i]);
      printf("************* double2float FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
    }
  }
  if(fabs(sum_ref - sum)>1e-4) {
      printf("WRONG at SUM: Ref:%lf vs SW:%lf\n",sum_ref,sum);
      printf("************* double2float FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
  }
  printf("double2float is OK.");
  free(src);
  free(dst);
  free(dst_ref);
#undef SRC_TYPE
#undef DST_TYPE
}

void test_float2double(int count,const char* n) {
#define SRC_TYPE float
#define DST_TYPE double
  printf("Test double to float conversion...\n");
  //int count = 32*1024*1024; // 8*32M double data;
  SRC_TYPE* src = (SRC_TYPE*)malloc(count*sizeof(SRC_TYPE));
  DST_TYPE* dst = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  DST_TYPE* dst_ref = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  printf("creating data... SIZE:%d\n",count);
  for( int i = 0; i < count; ++i )
    src[i] = rand()/(SRC_TYPE)RAND_MAX - 0.5;

  printf("calling sw_double2float.\n");
  char str1[100] = "Accelerated float2double ";
  strcat(str1,n);
  begin_timer(str1);
  float2double(src,dst,count);
  stop_timer();

  printf("calculating reference value.\n");
  char str2[100] = "Reference float2double ";
  strcat(str2,n);
  begin_timer(str2);
  for( int i = 0; i < count; ++i )
    dst_ref[i] = (DST_TYPE)src[i];
  stop_timer();

  printf("calculating errors...\n");
  float sum=0.0, sum_ref=0.0;
  printf("inout at 0: In:%lf Ref:%f vs SW:%f\n",src[0],dst_ref[0],dst[0]);
  for( int i = 0; i < count; ++i ) {
    sum+=dst[i];
    sum_ref+=dst_ref[i];
    if(fabs(dst_ref[i] - dst[i])>1e-4) {
      printf("WRONG at %d: Ref:%lf vs SW:%lf\n",i,dst_ref[i],dst[i]);
      printf("\tIn:%lf\n",src[i]);
      printf("************* double2float FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
    }
  }
  if(fabs(sum_ref - sum)>1e-4) {
      printf("WRONG at SUM: Ref:%lf vs SW:%lf\n",sum_ref,sum);
      printf("************* double2float FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
  }
  printf("double2float is OK.");
  free(src);
  free(dst);
  free(dst_ref);
#undef SRC_TYPE
#undef DST_TYPE
}

#define TEST_SIZE_1  64*1024
#define TEST_SIZE_2  128*1024
#define TEST_SIZE_3  256*1024
#define TEST_SIZE_4  512*1024
#define TEST_SIZE_5  1*1024*1024
#define TEST_SIZE_6  2*1024*1024
#define TEST_SIZE_7  4*1024*1024
#define TEST_SIZE_8  8*1024*1024
#define TEST_SIZE_9  16*1024*1024
#define TEST_SIZE_10 32*1024*1024

int main() {
  athread_init();
  //test_forward_pad();
  test_forward_pad_float();
  for(int i = 0; i < 10; ++i)
    test_forward_pad_float_fast();

  //test_backward_pad();
  //test_backward_pad_float();
  //test_backward_pad_float_fast();
  //test_backward_pad_split_float_fast();
  for(int i = 0; i < 10; ++i)
    test_backward_pad_split();

  //test_backward();
  //test_forward();
  //test_relu_forward();
  //test_relu_backward();
  //test_relu_forward_f();
  //test_relu_backward_f();

  /*
   * test data conversion
   */

  //test_double2float(TEST_SIZE_1,"1");
  //test_float2double(TEST_SIZE_1,"1");
  //test_double2float(TEST_SIZE_2,"2");
  //test_float2double(TEST_SIZE_2,"2");
  //test_double2float(TEST_SIZE_3,"3");
  //test_float2double(TEST_SIZE_3,"3");
  //test_double2float(TEST_SIZE_4,"4");
  //test_float2double(TEST_SIZE_4,"4");
  //test_double2float(TEST_SIZE_5,"5");
  //test_float2double(TEST_SIZE_5,"5");
  //test_double2float(TEST_SIZE_6,"6");
  //test_float2double(TEST_SIZE_6,"6");
  //test_double2float(TEST_SIZE_7,"7");
  //test_float2double(TEST_SIZE_7,"7");
  //test_double2float(TEST_SIZE_8,"8");
  //test_float2double(TEST_SIZE_8,"8");
  //test_double2float(TEST_SIZE_9,"9");
  //test_float2double(TEST_SIZE_9,"9");
  //test_double2float(TEST_SIZE_10,"10");
  //test_float2double(TEST_SIZE_10,"10");
  print_timer();
  return 0;
}
