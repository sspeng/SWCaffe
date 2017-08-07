#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
#include "caffe/swlayers/sw_conv_layer_impl.h"
#include "caffe/util/matrix_trans.h"
#include "caffe/util/data_type_trans.h"


extern SLAVE_FUN(conv_pad)();
extern SLAVE_FUN(conv_full_pad)();

// high -> low
// B, N, R, C
inline int image_caffe_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((n*C + c)*H + h)*W + w);
}
// R, C, N, B
inline int image_swdnn_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*C + c)*N + n);
}
// R, C, B, N
inline int image_swdnn_offset_back(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*N + n)*C + c);
}
// No, Ni, Kr, Kc
inline int weight_caffe_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return (( no*Ni + ni )*K + kr)*K + kc;
}
// Kr, Kc, No, Ni
inline int weight_swdnn_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*No + no) * Ni + ni );
}
// Kr, Kc, Ni, No
inline int weight_swdnn_offset_back(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*Ni + ni) * No + no );
}

//#define weight_swdnn_to_caffe(in,out,B,N,H,W) swapBN_HW(in,out,H,W,B,N)
//#define weight_caffe_to_swdnn(in,out,B,N,H,W) swapBN_HW(in,out,B,N,H,W)
//#define image_caffe_to_swdnn_back(in,out,B,N,H,W)  swapBN_HW(in,out,B,N,H,W)

typedef struct ConvData_st{
  void* input;  //0
  void* weight; //8
  void* output; //16
  //   24,  28,  32,  36, 40,  44,  48, 52, 56 
  int _Ni, _Ri, _Ci, _No, _K, _Ro, _Co, _B, _Costride, _bCo, _pad;
}ConvData;

static int init_flag = 0;

void sw_conv_forward_pad_impl_f_fast(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
	  printf("forward : before conv swDNN fast-float\n");
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    float* my_in      = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_weight  = (float*)malloc(sizeof(float)*K*K*No*Ni);
    double* my_out_d     = (double*)malloc(sizeof(double)*Ro*Co*No*B);

#ifdef MPE_TRANS
    printf("in_trans before");
    for(cRi = 0; cRi < Ri; ++cRi)
      for(cCi = 0; cCi < Ci; ++cCi)
        for(cNi = 0; cNi < Ni; ++cNi)
          for(cB = 0; cB < B; ++cB)
            my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
              in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
    printf("in_trans OVER");
#else
#ifdef SW_TRANS
    image_caffe_to_swdnn_f((float*)in,my_in,B,Ni,Ri,Ci);
#endif
#endif


#ifdef MPE_TRANS
    for(cNi = 0; cNi < Ni; ++cNi)
      for(cNo = 0; cNo < No; ++cNo)
        for(cKr = 0; cKr < K; ++cKr)
          for(cKc = 0; cKc < K; ++cKc)
              my_weight[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)] =
                weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
    printf("weight_trans OVER");
#else
#ifdef SW_TRANS
    weight_caffe_to_swdnn_f((float*)weight,my_weight,No,Ni,K,K);
#endif
#endif

    double* my_in_d      = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    float2double(my_in, my_in_d, Ri*Ci*Ni*B);
    free(my_in);
    double* my_weight_d  = (double*)malloc(sizeof(double)*K*K*No*Ni);
    float2double(my_weight, my_weight_d, K*K*No*Ni);
    free(my_weight);

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input =  my_in_d;
    param->weight = my_weight_d;
    param->output = my_out_d;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri+2*pad-K+1;
	  param->_Co = Ci+2*pad-K+1;
	  param->_B  = B;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*60*1024/8 - Ni*B-Ni*No)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No + No*B*Costride + Ni*B);
	  assert(ldm_consume < 64*1024*64);

	  athread_spawn(conv_pad, param);
	  athread_join();

    free(my_in_d);
    free(my_weight_d);

    float* my_out     = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    double2float(my_out_d, my_out, Ro*Co*No*B);
    free(my_out_d);

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
      for(cCo = 0; cCo < Co; ++cCo)
        for(cNo = 0; cNo < No; ++cNo)
          for(cB = 0; cB < B; ++cB)
            out[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] =
              my_out[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#else
#ifdef SW_TRANS
    image_swdnn_to_caffe_f(my_out,out,B,No,Ro,Co);
#endif
#endif
    free(my_out);
    free(param);
	  printf("forward : end conv swDNN fast-float\n");
}

//combine weigh_diff and top_grad updating in one function
void sw_conv_backward_pad_impl_f_fast(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
	  printf("begin Backward Pad Fast Float Impl\n");

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_in = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#else
#ifdef SW_TRANS
	  image_caffe_to_swdnn_back_f((float*)in,my_in,B, Ni, Ri, Ci);
#endif
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#else
#ifdef SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#endif
#endif

    //memset(my_weight_diff, 0, sizeof(float)*Ni*No*K*K);


    double* my_in_d = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    float2double(my_in, my_in_d, Ri*Ci*Ni*B);
    free(my_in);
    double* my_out_grad_d = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    float2double(my_out_grad, my_out_grad_d, Ro*Co*No*B);
    free(my_out_grad);
    double* my_weight_diff_d = (double*)malloc(sizeof(double)*Ni*No*K*K);

    param->input  = my_in_d;
    param->weight = my_out_grad_d;
    param->output = my_weight_diff_d;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
	  athread_spawn(conv_pad, param);
	  athread_join();

    float* my_weight_diff = (float*)malloc(sizeof(float)*Ni*No*K*K);
    double2float(my_weight_diff_d, my_weight_diff, Ni*No*K*K);
    free(my_weight_diff_d);


#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#else
#ifdef SW_TRANS
	  weight_swdnn_to_caffe_f(my_weight_diff, weight_diff,No, Ni, K, K);
#endif
#endif
	  printf("Backward weight_diff OK\n");

    free(my_weight_diff);
    free(my_in_d);

    //Transforamation and rot180 for Weight
    float* my_weight   = (float*)malloc(sizeof(float)*No*Ni*K*K);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#else
#ifdef SW_TRANS
	  weight_caffe_to_swdnn_back_f((float*)weight,my_weight,No, Ni, K, K);
#endif
#endif
    double* my_weight_d = (double*)malloc(sizeof(double)*No*Ni*K*K);
    float2double(my_weight, my_weight_d, No*Ni*K*K);
    free(my_weight);
    double* my_in_grad_d = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);

    param->input  =   my_out_grad_d;
    param->weight =   my_weight_d;
    param->output =   my_in_grad_d;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
	  //printf("Costride is %d\n", Costride);
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(float)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  athread_spawn(conv_full_pad, param);
    athread_join();

    float* my_in_grad = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    double2float(my_in_grad_d, my_in_grad, Ri*Ci*Ni*B);
    free(my_in_grad_d);

#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#else
#ifdef SW_TRANS
	  image_swdnn_to_caffe_f(my_in_grad,in_grad,B, Ni, Ri, Ci);
#endif
#endif
	  printf("Backward in_grad calc is OK!\n");

    free(my_in_grad);
    free(my_weight_d);
    free(my_out_grad_d);
    free(param);
}


void sw_conv_backward_pad_weight_diff_impl_f_fast(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
	  printf("begin Backward Pad Weight_Diff Fast Float Impl\n");

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_in = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#else
#ifdef SW_TRANS
	  image_caffe_to_swdnn_back_f((float*)in,my_in,B, Ni, Ri, Ci);
#endif
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#else
#ifdef SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#endif
#endif

    //memset(my_weight_diff, 0, sizeof(float)*Ni*No*K*K);


    double* my_in_d = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    float2double(my_in, my_in_d, Ri*Ci*Ni*B);
    free(my_in);
    double* my_out_grad_d = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    float2double(my_out_grad, my_out_grad_d, Ro*Co*No*B);
    free(my_out_grad);
    double* my_weight_diff_d = (double*)malloc(sizeof(double)*Ni*No*K*K);

    param->input  = my_in_d;
    param->weight = my_out_grad_d;
    param->output = my_weight_diff_d;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
	  athread_spawn(conv_pad, param);
	  athread_join();

    free(my_in_d);
    free(my_out_grad_d);

    float* my_weight_diff = (float*)malloc(sizeof(float)*Ni*No*K*K);
    double2float(my_weight_diff_d, my_weight_diff, Ni*No*K*K);
    free(my_weight_diff_d);


#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#else
#ifdef SW_TRANS
	  weight_swdnn_to_caffe_f(my_weight_diff, weight_diff,No, Ni, K, K);
#endif
#endif
    free(my_weight_diff);
    free(param);
	  printf("Backward weight_diff OK\n");
}

void sw_conv_backward_pad_in_diff_impl_f_fast(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
	  printf("begin Backward Pad in_Diff Fast Float Impl\n");
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    int Costride;

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_weight   = (float*)malloc(sizeof(float)*No*Ni*K*K);
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);
#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#else
#ifdef SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#endif
#endif

    double* my_out_grad_d = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    float2double(my_out_grad, my_out_grad_d, Ro*Co*No*B);
    free(my_out_grad);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#else
#ifdef SW_TRANS
	  weight_caffe_to_swdnn_back_f((float*)weight,my_weight,No, Ni, K, K);
#endif
#endif
    double* my_weight_d = (double*)malloc(sizeof(double)*No*Ni*K*K);
    float2double(my_weight, my_weight_d, No*Ni*K*K);
    free(my_weight);
    double* my_in_grad_d = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);

    param->input  =   my_out_grad_d;
    param->weight =   my_weight_d;
    param->output =   my_in_grad_d;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(float)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  athread_spawn(conv_full_pad, param);
    athread_join();

    free(my_weight_d);
    free(my_out_grad_d);

    float* my_in_grad = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    double2float(my_in_grad_d, my_in_grad, Ri*Ci*Ni*B);
    free(my_in_grad_d);

#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#else
#ifdef SW_TRANS
	  image_swdnn_to_caffe_f(my_in_grad,in_grad,B, Ni, Ri, Ci);
#endif
#endif
	  printf("Backward in_grad calc is OK!\n");

    free(my_in_grad);
    free(param);
}
