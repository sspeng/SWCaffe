extern "C" {
#include "caffe/swlayers/sw_relu_layer_impl.h"
}
#include "test_relu_layer_impl.hpp"
#include <math.h>
#include <stdio.h>
#include "athread.h"
#include "timer.h"

void test_relu_main() {
  //test_relu_forward();
  //test_relu_backward();
  //test_relu_forward_f();
  //test_relu_backward_f();
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


