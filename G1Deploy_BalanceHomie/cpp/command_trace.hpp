#pragma once
#include "protocol.hpp"
#include <algorithm>

namespace decoupled {
// Updated only AFTER Write succeeds. A failed/omitted write leaves the last
// successful command and its timestamp intact, including partial hand writes.
class CommandTrace {
 public:
  template<class Motors>
  void published(int group,const Motors& motors,uint64_t now) {
    const int offset=group==0?0:29+(group-1)*7;
    const int count=group==0?29:7;
    for(int j=0;j<count;++j) {
      q_[offset+j]=motors[j].q();
      const uint64_t bit=uint64_t(1)<<(offset+j);
      if(motors[j].kp()>0) position_mask_|=bit;
      else position_mask_&=~bit;
    }
    sent_ns_[group]=now;
  }
  void telemetry(StatePacket& packet,uint64_t now) const {
    packet.bridge_ns=now;
    packet.position_mask=position_mask_;
    std::copy(sent_ns_.begin(),sent_ns_.end(),packet.sent_ns);
    std::copy(q_.begin(),q_.end(),packet.sent_q);
  }
 private:
  std::array<uint64_t,3> sent_ns_{};
  uint64_t position_mask_=0;
  std::array<float,43> q_{};
};
}
