extern "C" {
#include "caffe/util/sw_memcpy.h"
}
#include <math.h>
#include <stdio.h>
#include "athread.h"
#include "timer.h"

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

void test_memcpy_main() {
  /*
  test_sw_memcpy_double(TEST_SIZE_1,"1");
  test_sw_memcpy_float (TEST_SIZE_1,"1");
  test_sw_memcpy_double(TEST_SIZE_2,"2");
  test_sw_memcpy_float (TEST_SIZE_2,"2");
  test_sw_memcpy_double(TEST_SIZE_3,"3");
  test_sw_memcpy_float (TEST_SIZE_3,"3");
  test_sw_memcpy_double(TEST_SIZE_4,"4");
  test_sw_memcpy_float (TEST_SIZE_4,"4");
  test_sw_memcpy_double(TEST_SIZE_5,"5");
  test_sw_memcpy_float (TEST_SIZE_5,"5");
  test_sw_memcpy_double(TEST_SIZE_6,"6");
  test_sw_memcpy_float (TEST_SIZE_6,"6");
  test_sw_memcpy_double(TEST_SIZE_7,"7");
  test_sw_memcpy_float (TEST_SIZE_7,"7");
  test_sw_memcpy_double(TEST_SIZE_8,"8");
  test_sw_memcpy_float (TEST_SIZE_8,"8");
  test_sw_memcpy_double(TEST_SIZE_9,"9");
  test_sw_memcpy_float (TEST_SIZE_9,"9");
  test_sw_memcpy_double(TEST_SIZE_10,"10");
  test_sw_memcpy_float (TEST_SIZE_10,"10");
  */

}

void test_sw_memcpy_double(int count,const char* n) {
#define SRC_TYPE double
#define DST_TYPE double
  printf("Test double memcpy...\n");
  //int count = 32*1024*1024; // 8*32M double data;
  SRC_TYPE* src = (SRC_TYPE*)malloc(count*sizeof(SRC_TYPE));
  DST_TYPE* dst = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  DST_TYPE* dst_ref = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  printf("creating data... SIZE:%d\n",count);
  for( int i = 0; i < count; ++i )
    src[i] = rand()/(SRC_TYPE)RAND_MAX - 0.5;

  printf("calling sw_double2float.\n");
  char str1[100] = "Accelerated double memcpy ";
  strcat(str1,n);
  begin_timer(str1);
  sw_memcpy_d(src,dst,count);
  stop_timer();

  printf("calculating reference value.\n");
  char str2[100] = "Reference double memcpy ";
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
      printf("************* memcpy_d FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
    }
  }
  if(fabs(sum_ref - sum)>1e-4) {
      printf("WRONG at SUM: Ref:%lf vs SW:%lf\n",sum_ref,sum);
      printf("************* memcpy_d FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
  }
  printf("memcpy_d is OK.");
  free(src);
  free(dst);
  free(dst_ref);
#undef SRC_TYPE
#undef DST_TYPE
}

void test_sw_memcpy_float(int count,const char* n) {
#define SRC_TYPE float
#define DST_TYPE float
  printf("Test float memcpy...\n");
  //int count = 32*1024*1024; // 8*32M double data;
  SRC_TYPE* src = (SRC_TYPE*)malloc(count*sizeof(SRC_TYPE));
  DST_TYPE* dst = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  DST_TYPE* dst_ref = (DST_TYPE*)malloc(count*sizeof(DST_TYPE));
  printf("creating data... SIZE:%d\n",count);
  for( int i = 0; i < count; ++i )
    src[i] = rand()/(SRC_TYPE)RAND_MAX - 0.5;

  printf("calling sw_double2float.\n");
  char str1[100] = "Accelerated float memcpy ";
  strcat(str1,n);
  begin_timer(str1);
  sw_memcpy_f(src,dst,count);
  stop_timer();

  printf("calculating reference value.\n");
  char str2[100] = "Reference float memcpy ";
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
      printf("************* memcpy_f FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
    }
  }
  if(fabs(sum_ref - sum)>1e-4) {
      printf("WRONG at SUM: Ref:%lf vs SW:%lf\n",sum_ref,sum);
      printf("************* memcpy_f FAILED ***************");
      free(src);
      free(dst);
      free(dst_ref);
      return ;
  }
  printf("memcpy_f is OK.");
  free(src);
  free(dst);
  free(dst_ref);
#undef SRC_TYPE
#undef DST_TYPE
}
