#include "command_trace.hpp"
#include <cassert>
using namespace decoupled;
struct Motor {
  float angle=0,gain=0;
  float q() const { return angle; }
  float kp() const { return gain; }
};
int main() {
  CommandTrace trace;
  StatePacket s;
  trace.telemetry(s,100);
  assert(s.bridge_ns==100 && s.sent_ns[0]==0 && s.position_mask==0);
  std::array<Motor,29> body;
  for(int i=0;i<29;++i) body[i]={float(i),10.f};
  body[13].gain=body[14].gain=0;
  trace.published(0,body,101);
  std::array<Motor,7> hand;
  for(int j=0;j<7;++j) hand[j]={float(j)*.1f,.5f};
  trace.published(1,hand,102);
  trace.telemetry(s,103);
  assert(s.sent_q[28]==28 && s.sent_q[35]==hand[6].angle && s.sent_ns[2]==0);
  assert((s.position_mask&(uint64_t(1)<<13))==0);
  // No successful right-hand Write: do not advance its trace or overwrite left.
  trace.telemetry(s,200);
  assert(s.sent_ns[0]==101 && s.sent_ns[1]==102 && s.sent_ns[2]==0);
  hand[0].angle=.7f;
  trace.published(2,hand,201);
  for(auto& motor:body) { motor.angle=0; motor.gain=0; }
  trace.published(0,body,202);
  trace.telemetry(s,203);
  assert(s.sent_q[36]==.7f && s.sent_ns[2]==201 && s.sent_ns[0]==202);
  assert((s.position_mask&((uint64_t(1)<<29)-1))==0);
  assert((s.position_mask&(uint64_t(1)<<42))!=0);
}
