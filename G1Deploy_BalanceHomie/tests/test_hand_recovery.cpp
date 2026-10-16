#include "hand_core.hpp"
#include <cassert>
#include <limits>
using namespace decoupled;

int main() {
  const uint64_t start=1000000000;
  const std::array<float,14> zero{};
  auto sample=[&] {
    HandFeedback f; f.valid.fill(true); f.received_ns.fill(start);
    f.q[8]=hand_lower[8]-.06f;
    return f;
  };
  auto f=sample();
  assert(f.healthy_flags(start)==3 && f.recovery_mask(start)==(1U<<8));
  assert(f.position_warning_mask()==(1U<<8));
  assert(f.problem(1,start).find("joint_out_of_bounds")!=std::string::npos);
  HandCore prepare_with_warning(true,zero);
  assert(prepare_with_warning.prepare(f,start)); // Position alone never blocks body/hand preparation.
  HandCore h(true,zero);
  assert(!h.request_recovery(8,Stage::Holding,f,start));
  assert(!h.request_recovery(7,Stage::Waiting,f,start));
  assert(h.request_recovery(8,Stage::Waiting,f,start));
  assert(h.recovering && !h.active && h.ever_enabled);
  assert(h.q[8]>hand_lower[8] && h.q[8]<hand_upper[8]);
  assert(!h.can_prepare(f,start));
  // A commanded in-range target is not completion. Fresh measured arrival is.
  for(int i=0;i<=100;++i) {
    auto now=start+i*20000000ULL;
    f.received_ns.fill(now);
    assert(!h.request_recovery(8,Stage::Waiting,f,now));
    assert(h.step_recovery(f,now).empty());
    assert(h.recovering && h.recovery_kp>=.5f && h.recovery_kp<=2.f);
    if(i==0) assert(h.recovery_kp==.5f);
    if(i==25) assert(h.recovery_kp==1.25f);
    if(i>=50) assert(h.recovery_kp==2.f);
  }
  f.q[8]=h.q[8];
  for(int i=101;i<=110;++i) {
    auto now=start+i*20000000ULL;
    f.received_ns.fill(now);
    h.request_recovery(8,Stage::Waiting,f,now);
    assert(h.step_recovery(f,now).empty());
  }
  assert(!h.recovering && !h.active && h.recovery_kp==0);
  assert(f.healthy_flags(start+2200000000ULL)==3 && f.position_warning_mask()==0);
  // A held command may not retrigger when the joint rebounds after completion.
  f.q[8]=hand_lower[8]-.06f;
  assert(!h.request_recovery(8,Stage::Waiting,f,start+2200000000ULL));
  h.request_recovery(-1,Stage::Waiting,f,start+2200000000ULL);
  assert(h.request_recovery(8,Stage::Waiting,f,start+2200000000ULL));
  h.step(start+2200000000ULL,true);
  assert(!h.recovering && h.recovery_joint==-1 && h.recovery_kp==0);

  for(int fault=0;fault<7;++fault) {
    auto bad=sample(); HandCore recovery(true,zero);
    assert(recovery.request_recovery(8,Stage::Waiting,bad,start));
    uint64_t now=start+20000000;
    bad.received_ns.fill(now);
    recovery.request_recovery(8,Stage::Waiting,bad,now);
    if(fault==0) bad.valid[1]=false; // Temperature / SDK motor error.
    if(fault==1) bad.q[8]=std::numeric_limits<float>::quiet_NaN();
    if(fault==2) bad.q[8]-=.04f; // Motion away from the target.
    if(fault==3) bad.dq[8]=.6f;
    if(fault==4) { now+=300000000; bad.received_ns.fill(now); } // Command lost.
    if(fault==5) bad.received_ns[1]=now-300000000; // DDS lost.
    if(fault==6) {
      now=start+3000000000ULL; bad.received_ns.fill(now);
      recovery.request_recovery(8,Stage::Waiting,bad,now);
    }
    auto warning=recovery.step_recovery(bad,now);
    assert(!warning.empty());
    assert(!recovery.recovering && !recovery.active && recovery.recovery_kp==0);
    assert(recovery.recovery_joint==-1 && recovery.dq==zero);
    assert(recovery.step_recovery(bad,now).empty()); // Warning emitted once.
    auto retry=sample(); retry.received_ns.fill(now);
    assert(!recovery.request_recovery(8,Stage::Waiting,retry,now)); // No automatic retry.
    if(fault==6) {
      assert(warning=="measured position did not recover in 3 s");
      assert(recovery.can_prepare(bad,now)); // Timeout cannot leave preparation stuck.
      recovery.request_recovery(-1,Stage::Waiting,retry,now);
      assert(recovery.request_recovery(8,Stage::Waiting,retry,now)); // Explicit retry still possible.
    }
  }
  // Upper-bound recovery also commands inward and times out at exactly 3 s.
  f=sample(); f.q=zero; f.q[6]=hand_upper[6]+.04f;
  HandCore upper(true,zero);
  assert(upper.request_recovery(6,Stage::Waiting,f,start));
  assert(upper.q[6]==hand_upper[6]-hand_recovery_inset);
  auto before_timeout=start+2995000000ULL;
  f.received_ns.fill(before_timeout);
  upper.request_recovery(6,Stage::Waiting,f,before_timeout);
  assert(upper.step_recovery(f,before_timeout).empty() && upper.recovering);
  f.received_ns.fill(start+3000000000ULL);
  assert(!upper.step_recovery(f,start+3000000000ULL).empty() && !upper.recovering);
  f=sample(); f.q[8]=hand_lower[8]-.2f;
  assert(f.recovery_mask(start)==0);
  f=sample(); f.valid[1]=false;
  assert(f.recovery_mask(start)==0);
  f=sample(); assert(f.recovery_mask(start+300000000)==0);
  // Compensation also applies to preparation only when explicitly requested.
  auto offsets=zero; offsets[7]=.73f;
  HandCore compensated(true,zero,offsets);
  f=sample(); f.q=zero;
  assert(compensated.prepare(f,start,3));
  compensated.step(start+3000000000ULL,false);
  assert(compensated.q==offsets && compensated.pose==3);
  HandCore raw(true,zero);
  assert(!raw.can_prepare(f,start,3));
}
