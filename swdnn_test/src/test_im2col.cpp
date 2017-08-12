extern "C" {
#include "athread.h"
#include "caffe/util/swim2col.h"
}
#include "test_im2col.hpp"
#include "caffe/util/im2col.hpp"
#include <math.h>
#include "timer.h"

void test_im2col_float(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n) {
#define Type float
  int dilation_h, dilation_w, stride_h, stride_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  stride_h = 1;
  stride_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  Type* data_col= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w+10*sizeof(Type));
  Type* data_col_ref= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w+10*sizeof(Type));

  printf("channels=%d, height=%d, width=%d, kernel_h=%d, kernel_w=%d, pad_h=%d, pad_w=%d, output_w=%d, output_h=%d\n",
          channels,height,width,kernel_h,kernel_w,pad_h,pad_w,output_w,output_h);
  printf("Now creating image data...\n");
  for( int i = 0; i < channels*height*width; ++i )
    data_im[i] = rand()/(Type)RAND_MAX;
  for(int i = 0; i<output_w*output_h*channels*kernel_h*kernel_w+10;++i) {
    data_col[i] = 2;
    data_col_ref[i] = 2;
  }
//#define PRINT_DATA
#ifdef PRINT_DATA
  printf("Created input data(at channel = 0):\n");
  for( int i = 0; i < height*width; ++i ) {
    if(i%width == 0) printf("\n\t");
    printf("%lf ",data_im[i]);
  }
  printf("\n");
#endif
  printf("Calling SW im2col...\n");
  char str1[100] = "swim2col_f ";
  strcat(str1,case_n);
  begin_timer(str1);
  swim2col_f(data_im,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col+5);
  stop_timer();
  printf("sw im2col is OK.\n");
#ifdef PRINT_DATA
  printf("SW output is:");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h)==0) printf("\n\t");
    printf("%lf ",data_col[i]);
  }
  printf("\n");
#endif
  printf("Calling caffe im2col...\n");
  char str2[100] = "Reference im2col_cpu ";
  strcat(str2,case_n);
  begin_timer(str2);
  caffe::im2col_cpu<Type>(data_im,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col_ref+5);
  stop_timer();
  printf("caffe im2col is OK.\n");
#ifdef PRINT_DATA
  printf("Caffe output is:");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h)==0) printf("\n\t");
    printf("%lf ",data_col_ref[i]);
  }
  printf("\n");
#endif
  printf("Now calculating Errors...\n");
  Type sum=0.0, sum_ref=0.0;
  int count = 0;
  int t_count = channels*output_w*output_h*kernel_h*kernel_w;
  for(int i = 0; i < channels*output_w*output_h*kernel_h*kernel_w+10;++i) {

    if(fabs(data_col_ref[i]-data_col[i])>1e-4) {
      count++;
      //if(count%126==1)
      printf("Wrong at (%d,%d,%d): Ref: %lf vs SW: %lf\n",
          i/(output_w*output_h*kernel_h*kernel_w),
          (i%(output_h*output_w*kernel_h*kernel_w))/(output_h*output_w),
          i%(output_w*output_h),data_col_ref[i],data_col[i]);
      printf("\tDEBUG INFO: %d, %d\n",
          i%(output_w*output_h)/output_w, i%(output_w*output_h)%output_w);
      /*
      printf("******************* swim2col_f FAILED **********************\n");
      free(data_im);
      free(data_col);
      free(data_col_ref);
      return ;*/
    }
    sum += data_col[i];
    sum_ref += data_col_ref[i];
  }
  printf("WA(%d in %d) Rate: %lf\n",count,t_count,count/(double)(channels*output_w*output_h*kernel_h*kernel_w));
  if(fabs(sum_ref-sum)>1e-4) {
    printf("Wrong at SUM: Ref: %lf vs SW: %lf\n",sum_ref,sum);
    printf("******************* swim2col_f FAILED *********************");
    free(data_im);
    free(data_col);
    free(data_col_ref);
    return ;
  }
  printf("Sum Ref: %lf vs SW: %lf\n",sum_ref,sum);
  printf("swim2col float test passed.\n");

#undef Type
}

void test_im2col_double(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n) {
#define Type double
  int dilation_h, dilation_w, stride_h, stride_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  stride_h = 1;
  stride_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  Type* data_col= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w);
  Type* data_col_ref= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w);

  printf("channels=%d, height=%d, width=%d, kernel_h=%d, kernel_w=%d, pad_h=%d, pad_w=%d, output_w=%d, output_h=%d\n",
          channels,height,width,kernel_h,kernel_w,pad_h,pad_w,output_w,output_h);
  printf("Now creating image data...\n");
  for( int i = 0; i < channels*height*width; ++i )
    data_im[i] = rand()/(double)RAND_MAX;
//#define PRINT_DATA
#ifdef PRINT_DATA
  printf("Created input data(at channel = 0):\n");
  for( int i = 0; i < height*width; ++i ) {
    if(i%width == 0) printf("\n\t");
    printf("%lf ",data_im[i]);
  }
  printf("\n");
#endif
  printf("Calling SW im2col...\n");
  char str1[100] = "swim2col_d ";
  strcat(str1,case_n);
  begin_timer(str1);
  swim2col_d(data_im,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col);
  stop_timer();
  printf("sw im2col is OK.\n");
#ifdef PRINT_DATA
  printf("SW output is:");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h)==0) printf("\n\t");
    printf("%lf ",data_col[i]);
  }
  printf("\n");
#endif
  printf("Calling caffe im2col...\n");
  char str2[100] = "Reference im2col_cpu ";
  strcat(str2,case_n);
  begin_timer(str2);
  caffe::im2col_cpu<double>(data_im,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col_ref);
  stop_timer();
  printf("caffe im2col is OK.\n");
#ifdef PRINT_DATA
  printf("Caffe output is:");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h)==0) printf("\n\t");
    printf("%lf ",data_col_ref[i]);
  }
  printf("\n");
#endif
  printf("Now calculating Errors...\n");
  double sum=0.0, sum_ref=0.0;
  int count = 0;
  int t_count = channels*output_w*output_h*kernel_h*kernel_w;
  for(int i = 0; i < channels*output_w*output_h*kernel_h*kernel_w;++i) {

    if(fabs(data_col_ref[i]-data_col[i])>1e-4) {
      count++;
      //if(count%126==1)
      printf("Wrong at (%d,%d,%d): Ref: %lf vs SW: %lf\n",
          i/(output_w*output_h*kernel_h*kernel_w),
          (i%(output_h*output_w*kernel_h*kernel_w))/(output_h*output_w),
          i%(output_w*output_h),data_col_ref[i],data_col[i]);
      printf("\tDEBUG INFO: %d, %d\n",
          i%(output_w*output_h)/output_w, i%(output_w*output_h)%output_w);
      /*
      printf("******************* swim2col FAILED **********************\n");
      free(data_im);
      free(data_col);
      free(data_col_ref);
      return ;*/
    }
    sum += data_col[i];
    sum_ref += data_col_ref[i];
  }
  printf("WA(%d in %d) Rate: %lf\n",count,t_count,count/(double)(channels*output_w*output_h*kernel_h*kernel_w));
  if(fabs(sum_ref-sum)>1e-4) {
    printf("Wrong at SUM: Ref: %lf vs SW: %lf\n",sum_ref,sum);
    printf("******************* swim2col FAILED *********************");
    free(data_im);
    free(data_col);
    free(data_col_ref);
    return ;
  }
  printf("Sum Ref: %lf vs SW: %lf\n",sum_ref,sum);
  printf("swim2col double test passed.\n");

#undef Type
}

void test_col2im_float(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n) {
#define Type float
  int dilation_h, dilation_w, stride_h, stride_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  stride_h = 1;
  stride_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  Type* data_col= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w);
  Type* data_im_ref= (Type*)malloc(sizeof(Type)*channels*height*width);

  printf("channels=%d, height=%d, width=%d, kernel_h=%d, kernel_w=%d, pad_h=%d, pad_w=%d, output_w=%d, output_h=%d\n",
          channels,height,width,kernel_h,kernel_w,pad_h,pad_w,output_w,output_h);
  printf("Now creating image data...\n");
  for( int i = 0; i < channels*output_w*output_h*kernel_h*kernel_w; ++i )
    data_col[i] = rand()/(float)RAND_MAX;
#ifdef PRINT_DATA
  printf("Created input data(at channel = 0):\n");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h) == 0) printf("\n\t");
    printf("%lf ",data_col[i]);
  }
  printf("\n");
#endif
  printf("Calling SW col2im...\n");
  char str1[100] = "swcol2im_f ";
  strcat(str1,case_n);
  begin_timer(str1);
  swcol2im_f(data_col,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_im);
  stop_timer();
  printf("sw col2im is OK.\n");
#ifdef PRINT_DATA
  printf("SW output is:");
  for( int i = 0; i < height*width; ++i ) {
    if(i%(width)==0) printf("\n\t");
    printf("%lf ",data_im[i]);
  }
  printf("\n");
#endif
  printf("Calling caffe col2im...\n");
  char str2[100] = "Reference col2im_cpu ";
  strcat(str2,case_n);
  begin_timer(str2);
  caffe::col2im_cpu<float>(data_col,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_im_ref);
  stop_timer();
  printf("caffe col2im is OK.\n");
#ifdef PRINT_DATA
  printf("Caffe output is:");
  for( int i = 0; i < width*height; ++i ) {
    if(i%(width)==0) printf("\n\t");
    printf("%lf ",data_im_ref[i]);
  }
  printf("\n");
#endif
  printf("Now calculating Errors...\n");
  float sum=0.0, sum_ref=0.0;
  int count = 0;
  int t_count = channels*height*width;
  for(int i = 0; i < channels*height*width;++i) {
    if(fabs(data_im_ref[i]-data_im[i])>1e-4) {
      count++;
#ifndef PRINT_DATA
      //if((i/(height*width))<2 && i%width==0)
      printf("Wrong at (%d,%d,%d): Ref: %f vs SW: %f\n",
          i/(height*width),
          (i%(height*width))/(width),
          i%(width),data_im_ref[i],data_im[i]);
      //printf("\tDEBUG INFO: %d, %d\n",
      //    i%(output_w*output_h)/output_w, i%(output_w*output_h)%output_w);
      /*
      printf("******************* swim2col FAILED **********************\n");
      free(data_im);
      free(data_col);
      free(data_col_ref);
      return ;*/
#endif
    }
    sum += data_im[i];
    sum_ref += data_im_ref[i];
  }
  printf("WA(%d in %d) Rate: %lf\n",count,t_count,count/(double)(channels*height*width));
  if(fabs(sum_ref-sum)>1e-4) {
    printf("Wrong at SUM: Ref: %lf vs SW: %lf\n",sum_ref,sum);
    printf("******************* swcol2im_f FAILED *********************");
    free(data_im);
    free(data_col);
    free(data_im_ref);
    return ;
  }
  printf("Sum Ref: %lf vs SW: %lf\n",sum_ref,sum);
  printf("swcol2im float test passed.\n");
  free(data_im);
  free(data_col);
  free(data_im_ref);
#undef Type
}

void test_col2im_double(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n) {
#define Type double
  int dilation_h, dilation_w, stride_h, stride_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  stride_h = 1;
  stride_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  Type* data_col= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w);
  Type* data_im_ref= (Type*)malloc(sizeof(Type)*channels*height*width);

  printf("channels=%d, height=%d, width=%d, kernel_h=%d, kernel_w=%d, pad_h=%d, pad_w=%d, output_w=%d, output_h=%d\n",
          channels,height,width,kernel_h,kernel_w,pad_h,pad_w,output_w,output_h);
  printf("Now creating image data...\n");
  for( int i = 0; i < channels*output_w*output_h*kernel_h*kernel_w; ++i )
    data_col[i] = rand()/(double)RAND_MAX;
#ifdef PRINT_DATA
  printf("Created input data(at channel = 0):\n");
  for( int i = 0; i < output_w*output_h*kernel_h*kernel_w; ++i ) {
    if(i%(output_w*output_h) == 0) printf("\n\t");
    printf("%lf ",data_col[i]);
  }
  printf("\n");
#endif
  printf("Calling SW col2im...\n");
  char str1[100] = "swcol2im_d ";
  strcat(str1,case_n);
  begin_timer(str1);
  swcol2im_d(data_col,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_im);
  stop_timer();
  printf("sw col2im is OK.\n");
#ifdef PRINT_DATA
  printf("SW output is:");
  for( int i = 0; i < height*width; ++i ) {
    if(i%(width)==0) printf("\n\t");
    printf("%lf ",data_im[i]);
  }
  printf("\n");
#endif
  printf("Calling caffe col2im...\n");
  char str2[100] = "Reference col2im_cpu ";
  strcat(str2,case_n);
  begin_timer(str2);
  caffe::col2im_cpu<double>(data_col,channels,height,width,kernel_h,kernel_w,
              pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_im_ref);
  stop_timer();
  printf("caffe col2im is OK.\n");
#ifdef PRINT_DATA
  printf("Caffe output is:");
  for( int i = 0; i < width*height; ++i ) {
    if(i%(width)==0) printf("\n\t");
    printf("%lf ",data_im_ref[i]);
  }
  printf("\n");
#endif
  printf("Now calculating Errors...\n");
  double sum=0.0, sum_ref=0.0;
  int count = 0;
  int t_count = channels*height*width;
  for(int i = 0; i < channels*height*width;++i) {
    if(fabs(data_im_ref[i]-data_im[i])>1e-4) {
      count++;
#ifndef PRINT_DATA
      //if((i/(height*width))<2 && i%width==0)
      printf("Wrong at (%d,%d,%d): Ref: %lf vs SW: %lf\n",
          i/(height*width),
          (i%(height*width))/(width),
          i%(width),data_im_ref[i],data_im[i]);
      //printf("\tDEBUG INFO: %d, %d\n",
      //    i%(output_w*output_h)/output_w, i%(output_w*output_h)%output_w);
      /*
      printf("******************* swim2col FAILED **********************\n");
      free(data_im);
      free(data_col);
      free(data_col_ref);
      return ;*/
#endif
    }
    sum += data_im[i];
    sum_ref += data_im_ref[i];
  }
  printf("WA(%d in %d) Rate: %lf\n",count,t_count,count/(double)(channels*height*width));
  if(fabs(sum_ref-sum)>1e-4) {
    printf("Wrong at SUM: Ref: %lf vs SW: %lf\n",sum_ref,sum);
    printf("******************* swcol2im_d FAILED *********************");
    free(data_im);
    free(data_col);
    free(data_im_ref);
    return ;
  }
  printf("Sum Ref: %lf vs SW: %lf\n",sum_ref,sum);
  printf("swcol2im double test passed.\n");
  free(data_im);
  free(data_col);
  free(data_im_ref);
#undef Type
}

void test_im2col_main() {
  /*
  test_col2im_float(IM2COL_TEST_CASE);
  test_col2im_float(IM2COL_TEST_CASE_0);
  test_col2im_float(IM2COL_TEST_CASE_1);
  test_col2im_float(IM2COL_TEST_CASE_2);
  test_col2im_float(IM2COL_TEST_CASE_3);
  test_col2im_float(IM2COL_TEST_CASE_4);
  test_col2im_float(IM2COL_TEST_CASE_5);
  test_col2im_float(IM2COL_TEST_CASE_6);
  test_col2im_float(IM2COL_TEST_CASE_7);
  test_col2im_float(IM2COL_TEST_CASE_8);
  test_col2im_float(IM2COL_TEST_CASE_9);
*/
/*
  test_col2im_double(IM2COL_TEST_CASE);
  test_col2im_double(IM2COL_TEST_CASE_0);
  test_col2im_double(IM2COL_TEST_CASE_1);
  test_col2im_double(IM2COL_TEST_CASE_2);
  test_col2im_double(IM2COL_TEST_CASE_3);
  test_col2im_double(IM2COL_TEST_CASE_4);
  test_col2im_double(IM2COL_TEST_CASE_5);
  test_col2im_double(IM2COL_TEST_CASE_6);
  test_col2im_double(IM2COL_TEST_CASE_7);
  test_col2im_double(IM2COL_TEST_CASE_8);
  */

  //test_im2col_float(IM2COL_TEST_CASE_1);
  //test_im2col_float(IM2COL_TEST_CASE_2);
  //test_im2col_float(IM2COL_TEST_CASE_3);
  //test_im2col_float(IM2COL_TEST_CASE_4);
  //test_im2col_float(IM2COL_TEST_CASE_5);
  //test_im2col_float(IM2COL_TEST_CASE_6);
  test_im2col_float(IM2COL_TEST_CASE_7);
  test_im2col_float(IM2COL_TEST_CASE_8);

/*
  test_im2col_double(IM2COL_TEST_CASE_1);
  test_im2col_double(IM2COL_TEST_CASE_2);
  test_im2col_double(IM2COL_TEST_CASE_3);
  test_im2col_double(IM2COL_TEST_CASE_4);
  test_im2col_double(IM2COL_TEST_CASE_5);
  test_im2col_double(IM2COL_TEST_CASE_6);
  test_im2col_double(IM2COL_TEST_CASE_7);
  test_im2col_double(IM2COL_TEST_CASE_8);
*/
}
