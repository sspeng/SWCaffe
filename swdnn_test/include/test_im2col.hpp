#ifndef TEST_IM2COL_H_
#define TEST_IM2COL_H_
//#define PRINT_DATA

#define IM2COL_TEST_CASE 1,5,5,3,3,1,1,"DEBUG CASE 1" // default test case
#define IM2COL_TEST_CASE_0 26,5,5,3,3,1,1,"DEBUG CASE 2" // default test case
#define IM2COL_TEST_CASE_1 128,12,12,3,3,1,1, "Case 1"
#define IM2COL_TEST_CASE_2 3,7,7,3,3,1,1, "Case 2" // odd size with padding
#define IM2COL_TEST_CASE_3 3,7,7,3,3,0,0, "Case 3" // odd size with no padding
#define IM2COL_TEST_CASE_4 22,128,128,4,4,0,0, "Case 4"
#define IM2COL_TEST_CASE_5 22,128,128,4,4,1,1, "Case 5"
#define IM2COL_TEST_CASE_6 22,128,128,4,4,2,2, "Case 6"
#define IM2COL_TEST_CASE_7 3 ,224,224,3,3,1,1, "Case 7"
#define IM2COL_TEST_CASE_8 64,224,224,3,3,1,1, "Case 8"
#define IM2COL_TEST_CASE_9 22,128,128,2,2,2,2, "Case 9" // 2x2 kernel

void test_im2col_float(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n);

void test_im2col_double(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n);

void test_col2im_float(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n);

void test_col2im_double(int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, const char* case_n);

void test_im2col_main();

#endif
