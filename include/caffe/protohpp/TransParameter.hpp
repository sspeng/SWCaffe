#ifndef _TransPARAMETER_H_
#define _TransPARAMETER_H_

namespace caffe {

class TransParameter {
public:

  TransParameter(){}
  explicit TransParameter(int B, int N, int R, int C,
      bool type):B_(B), N_(N), R_(R), C_(C), type_(type_){}

  TransParameter(const ReLUParameter& other){
    this->CopyFrom(other);
  }

  inline TransParameter& operator=(const ReLUParameter& other) {
    this->CopyFrom(other);
    return *this;
  }

  void CopyFrom(const TransParameter& other) {
    B_ = other.get_B();
    N_ = other.get_N();
    C_ = other.get_C();
    R_ = other.get_R();
    type_ = other.get_type();
  }

  inline void set_B (int B) { B_ = B; }
  inline void set_N (int N) { N_ = N; }
  inline void set_C (int C) { C_ = C; }
  inline void set_R (int R) { R_ = R; }
  inline void set_type (bool type) { type_ = type; }
  inline int get_B () const { return B_; }
  inline int get_N () const { return N_; }
  inline int get_C () const { return C_; }
  inline int get_R () const { return R_; }
  inline bool get_type () const { return type_; }

private:
  int B_, N_, R_, C_;
  //true blas-swdnn
  //false swdnn-blas
  bool type_;
};

}
#endif
