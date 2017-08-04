#ifndef _PReLUParameter_
#define _PReLUParameter_

#include "FillerParameter.hpp"

namespace caffe {

enum PReLUParameter_Engine {
  PReLUParameter_Engine_DEFAULT = 0,
  PReLUParameter_Engine_CAFFE = 1,
  PReLUParameter_Engine_CUDNN = 2
};

class PReLUParameter {
  public:
    PReLUParameter():engine_(PReLUParameter_Engine_DEFAULT)
	{
      channel_shared_ = false;
	  has_filler_ = false;
    }
    
    PReLUParameter(const PReLUParameter& other){
      this->CopyFrom(other);
    }

    inline PReLUParameter& operator=(const PReLUParameter& other) {
      this->CopyFrom(other);
      return *this;
    }
    
    inline bool channel_shared() const { return channel_shared_; }
    inline bool has_filler() const { return has_filler_; }
	inline PReLUParameter_Engine engine() const { return engine_; }

    inline void set_channel_shared(bool g) { channel_shared_ = g; }
	inline const FillerParameter& filler() const { return filler_; }
    inline FillerParameter* mutable_filler() { has_filler_ = true; return &filler_; }
    inline void set_engine (PReLUParameter_Engine value ) { engine_ = value; }

    void CopyFrom( const PReLUParameter& other ) {
      channel_shared_ = other.channel_shared();
      engine_ = other.engine();
    }
    static PReLUParameter default_instance_;
  private:
    bool channel_shared_;
	FillerParameter filler_;
    bool has_filler_;
    PReLUParameter_Engine engine_;
};

}
#endif
