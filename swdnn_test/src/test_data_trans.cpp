extern "C" {
#include "caffe/util/data_type_trans.h"
}
#include "athread.h"
#include <math.h>
#include "timer.h"
#include "test_data_trans.hpp"

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

void test_data_trans_main() {
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
  //
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
