#include "caffe/caffe.hpp"
#ifdef MYMPI
#include <mpi.h>
#endif
#include <string>
using namespace caffe;

int main (int argc, char ** argv) {
#ifdef MYMPI
  MPI_Init(&argc, &argv);
#endif

  SolverParameter solver_param;
  solver_param.add_test_iter(1);
  solver_param.set_test_interval(50);
  solver_param.set_base_lr(0.01);
  solver_param.set_display(10);
  solver_param.set_max_iter(450000);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(0.1);
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_type("SGD");


  NetParameter net_param = solver_param.mutable_net_param();
  LayerParameter* tmplp = net_param.add_layer();
  tmplp->set_name("data");
  tmplp->set_type("IMAGENETData");
  tmplp->add_top("data");
  tmplp->add_top("label");
  //TODO I double not like write it this way, No mirror crop_size 
  DataParameter* tmpdata_param = tmplp->add_data_param(); 
  NetStateRule train_include;
  train_include.set_phase(TRAIN);
  tmplp->add_include(train_include);
  tmpdata_param->set_source("../data/imagenet_bin/train_data.bin", "../data/imagenet_bin/train_label.bin", "../data/imagenet/train_mean.bin");
  tmpdata_param->set_batch_size(128);

  tmplp = net_param.add_layer();
  tmplp->set_name("data");
  tmplp->set_type("IMAGENETData");
  tmplp->add_top("data");
  tmplp->add_top("label");
  //I double not like write it this way
  DataParameter* tmpdata_param = tmplp->add_data_param(); 
  NetStateRule test_include;
  train_include.set_phase(TEST);
  tmplp->add_include(test_include);
  tmpdata_param->set_source("../data/imagenet_bin/test_data.bin", "../data/imagenet_bin/test_label.bin", "../data/imagenet/test_mean.bin");
  tmpdata_param->set_batch_size(50);

  ParamSpec * tmpps;
  ConvolutionParameter* tmpconvp;
  FillerParameter* fillerp;

  //1st conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv1");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("data");
  tmplp->add_top("conv1");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp->set_num_output(96);
  tmpconvp->set_kernel_size(11);
  tmpconvp->set_stride(4);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu1");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv1");
  tmplp->add_top("conv1");

  tmplp = net_param.add_layer();
  tmplp->set_name("norm1");
  tmplp->set_type("LRN");
  tmplp->add_bottom("conv1");
  tmplp->add_top("norm1");
  tmplp->set_local_size(5);
  tmplp->set_alpha(0.0001);
  tmplp->set_beta(0.75);

  PoolingParameter* tmppoolingp;
  tmplp = net_param.add_layer();
  tmplp->set_name("pool1");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("norm1");
  tmplp->add_top("pool1");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);


  //2nd conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv2");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("pool1");
  tmplp->add_top("conv2");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp->set_num_output(256);
//tmpconvp->set_pad(2);
  tmpconvp->set_kernel_size(5);
//  tmpconvp->set_stride(14);
//  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu2");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv2");
  tmplp->add_top("conv2");

  //TODO
  tmplp = net_param.add_layer();
  tmplp->set_name("norm2");
  tmplp->set_type("LRN");
  tmplp->add_bottom("conv2");
  tmplp->add_top("norm2");
  tmplp->set_local_size(5);
  tmplp->set_alpha(0.0001);
  tmplp->set_beta(0.75);

  PoolingParameter* tmppoolingp;
  tmplp = net_param.add_layer();
  tmplp->set_name("pool2");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("norm1");
  tmplp->add_top("pool2");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);


  //3rd conv-lrn-relu-pool layer
  tmplp = net_param.add_layer();
  tmplp->set_name("conv3");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("pool2");
  tmplp->add_top("conv3");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp->set_num_output(384);
//  tmpconvp->set_pad(1);
  tmpconvp->set_kernel_size(3);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu3");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv3");
  tmplp->add_top("conv3");

  //4nd conv
  tmplp = net_param.add_layer();
  tmplp->set_name("conv4");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("conv3");
  tmplp->add_top("conv4");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp->set_num_output(384);
//  tmpconvp->set_pad(1);
  tmpconvp->set_kernel_size(3);
//  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu4");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv4");
  tmplp->add_top("conv4");

  //5nd conv + relu
  tmplp = net_param.add_layer();
  tmplp->set_name("conv5");
  tmplp->set_type("Convolution");
  tmplp->add_bottom("conv4");
  tmplp->add_top("conv5");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpconvp->set_num_output(256);
//  tmpconvp->set_pad(1);
  tmpconvp->set_kernel_size(3);
//  tmpconvp->set_group(2);
  fillerp = tmpconvp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpconvp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu5");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("conv5");
  tmplp->add_top("conv5");

  tmplp = net_param.add_layer();
  tmplp->set_name("pool5");
  tmplp->set_type("Pooling");
  tmplp->add_bottom("conv5");
  tmplp->add_top("pool5");
  tmppoolingp = tmplp->add_pooling_param();
  tmppoolingp->set_pool(MAX);
  tmppoolingp->set_kernel_size(3);
  tmppoolingp->set_stride(2);

  //layer 6
InnerProductParameter* tmpipp;
  tmplp = net_param.add_layer();
  tmplp->set_name("fc6");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("pool5");
  tmplp->add_top("fc6");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(4096);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.005);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu6");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc6");

  tmplp = net_param.add_layer();
  tmplp->set_name("drop6");
  tmplp->set_type("Dropout");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc6");
  DropoutParameter* dropp = tmplp->add_dropout_param();
  dropp->set_dropout_ratio(0.5);



  //layer 7
  InnerProductParameter* tmpipp;
  tmplp = net_param.add_layer();
  tmplp->set_name("fc7");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("fc6");
  tmplp->add_top("fc7");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(4096);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.005);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0.1);

  tmplp = net_param.add_layer();
  tmplp->set_name("relu7");
  tmplp->set_type("ReLU");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc7");

  tmplp = net_param.add_layer();
  tmplp->set_name("drop7");
  tmplp->set_type("Dropout");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc7");
  DropoutParameter* dropp = tmplp->add_dropout_param();
  dropp->set_dropout_ratio(0.5);

  //8th layer
  tmplp = net_param.add_layer();
  tmplp->set_name("fc8");
  tmplp->set_type("InnerProduct");
  tmplp->add_bottom("fc7");
  tmplp->add_top("fc8");
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(1);
  tmpps->set_decay_mult(1);
  tmpps = tmplp->add_param();
  tmpps->set_lr_mult(2);
  tmpps->set_decay_mult(0);
  tmpipp = tmplp->add_inner_product_param(); 
  tmpipp->set_num_output(16);
  fillerp = tmpipp->mutable_weight_filler();
  fillerp->set_type("gaussian");
  fillerp->set_std(0.01);
  fillerp = tmpipp->mutable_bias_filler();
  fillerp->set_type("constant");
  fillerp->set_value(0);

  //layer 8
  tmplp = net_param.add_layer();
  tmplp->set_name("accuracy");
  tmplp->set_type("Accuracy");
  tmplp->add_bottom("fc8");
  tmplp->add_bottom("label");
  tmplp->add_top("accuracy");
  tmplp->add_include();
  tmplp->add_include(test_include);

  //layer loss
  tmplp = net_param.add_layer();
  tmplp->set_name("loss");
  tmplp->set_type("SoftmaxWithLoss");
  tmplp->add_bottom("fc8");
  tmplp->add_bottom("label");
  tmplp->add_top("loss");

  DLOG(INFO) << "Init solver...";
  shared_ptr<Solver<double> >
      solver(SolverRegistry<double>::CreateSolver(solver_param));
  DLOG(INFO) << "Begin solve...";
  //solver->net()->CopyTrainedLayersFrom("_iter_700.caffemodel");
  solver->Solve(NULL);
  LOG_IF(INFO, Caffe::root_solver())
    << "test end";
#ifdef MYMPI
  MPI_Finalize();
#endif
  return 0;
}
