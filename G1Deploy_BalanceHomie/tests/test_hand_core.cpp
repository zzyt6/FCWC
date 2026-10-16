#include "hand_core.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace decoupled;
int main() {
  const std::array<float,14> zeros{};
  const auto grasp=[] { auto q=hand_lower; q[0]=q[7]=0;
    q[1]=.335606f; q[2]=1.541224f; q[8]=-.333515f; q[9]=-1.544997f;
    for(int i=10;i<14;++i) q[i]=hand_upper[i];
    return q; }();
  HandCore hands(true,grasp);
  HandFeedback feedback;
  uint64_t now=1000000000;
  assert(!hands.prepare(feedback,now));
  assert(!hands.ever_enabled);
  feedback.valid.fill(true); feedback.received_ns.fill(now); feedback.q=grasp;
  assert(hands.prepare(feedback,now));
  hands.step(now,false);
  assert(hands.q==grasp && hands.dq==zeros);
  auto previous=hands.q;
  for(int tick=1;tick<=600;++tick) {
    now+=5000000;
    hands.step(now,false);
    for(int i=0;i<14;++i) {
      assert(std::abs(hands.q[i])<=std::abs(previous[i])+1e-6);
      assert(hands.q[i]*grasp[i]>=-1e-6);
      assert(std::abs(hands.q[i]-previous[i])<.006);
    }
    previous=hands.q;
    if(tick==300) for(int i=0;i<14;++i) assert(std::abs(hands.q[i]-grasp[i]/2)<1e-6);
    if(tick<600) assert(hands.moving);
  }
  assert(!hands.moving && hands.progress==1);
  assert(hands.q==zeros && hands.dq==zeros);
  hands.request(2,Stage::Waiting,now); assert(!hands.moving);
  hands.request(2,Stage::Holding,now); assert(hands.moving && hands.pose==2);
  hands.step(now,false); assert(hands.q==zeros);
  hands.step(now+1500000000,false);
  for(int i=0;i<14;++i) assert(std::abs(hands.q[i]-grasp[i]/2)<1e-6);
  hands.request(1,Stage::Policy,now+1500000000); assert(hands.pose==2);
  hands.step(now+3000000000,false);
  assert(hands.q==grasp && !hands.moving);
  // A held level command cannot repeatedly restart the finished ramp.
  hands.request(2,Stage::Policy,now+3100000000); assert(!hands.moving);
  hands.request(1,Stage::Policy,now+3100000000); assert(hands.moving);
  hands.step(now+4000000000,true);
  assert(!hands.active && !hands.moving && hands.dq==zeros);
  previous=hands.q;
  hands.request(2,Stage::Policy,now+4500000000);
  hands.step(now+6000000000,false);
  assert(hands.q==previous && !hands.active);
  feedback.received_ns.fill(now);
  assert(feedback.healthy_flags(now)==3);
  assert(feedback.healthy_flags(now+200000001)==0);
  feedback.valid[0]=false; assert(feedback.healthy_flags(now)==2);
  feedback.valid[0]=true; feedback.q[0]=std::numeric_limits<float>::quiet_NaN();
  assert(feedback.healthy_flags(now)==2);
  auto bad=grasp; bad[8]=10;
  bool rejected=false;
  try { HandCore invalid(true,bad); } catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
  auto compensated_zero=zeros; compensated_zero[7]=.7f;
  HandCore calibrated(true,grasp,compensated_zero);
  feedback.q=zeros; feedback.valid.fill(true); feedback.received_ns.fill(now);
  assert(calibrated.prepare(feedback,now));
  calibrated.step(now+3000000000,false);
  assert(calibrated.q==zeros); // Preparation never applies compensation.
  now+=3000000000;
  calibrated.request(3,Stage::Waiting,now); assert(!calibrated.moving);
  calibrated.request(3,Stage::Holding,now);
  calibrated.step(now,false); assert(calibrated.q==zeros && calibrated.dq==zeros);
  previous=calibrated.q;
  for(int tick=1;tick<=600;++tick) {
    calibrated.step(now+tick*5000000ULL,false);
    if(tick<600) {
      calibrated.request(2,Stage::Holding,now+tick*5000000ULL);
      assert(calibrated.pose==3); // Busy requests cannot change the target.
    }
    assert(calibrated.q[7]>=previous[7]);
    assert(calibrated.q[7]-previous[7]<.003f);
    for(int j=0;j<14;++j) if(j!=7) assert(calibrated.q[j]==0);
    if(tick==300) assert(std::abs(calibrated.q[7]-compensated_zero[7]/2)<1e-6);
    previous=calibrated.q;
  }
  assert(calibrated.q==compensated_zero && calibrated.dq==zeros && !calibrated.moving);
  StatePacket hand_packet;
  calibrated.telemetry(hand_packet,feedback,now);
  assert(hand_packet.hand_flags & 64);
  assert(!(hand_packet.hand_flags & 16));
  now+=3000000000;
  calibrated.request(3,Stage::Holding,now); assert(!calibrated.moving); // No accumulation.
  calibrated.request(4,Stage::Waiting,now); assert(!calibrated.moving);
  calibrated.request(4,Stage::Holding,now);
  calibrated.step(now,false); assert(calibrated.q==compensated_zero && calibrated.dq==zeros);
  for(int tick=1;tick<=600;++tick) {
    previous=calibrated.q;
    calibrated.step(now+tick*5000000ULL,false);
    assert(calibrated.q[7]==.7f); // Compensation remains constant while closing.
    for(int i=0;i<14;++i) assert(std::abs(calibrated.q[i]-previous[i])<.006f);
    if(tick<600) {
      calibrated.request(3,Stage::Holding,now+tick*5000000ULL);
      assert(calibrated.pose==4);
    }
  }
  auto compensated_grasp=grasp;
  for(int i=0;i<14;++i) compensated_grasp[i]+=compensated_zero[i];
  assert(calibrated.q==compensated_grasp && calibrated.dq==zeros && !calibrated.moving);
  calibrated.telemetry(hand_packet,feedback,now);
  assert((hand_packet.hand_flags & 80)==80);
  now+=3000000000;
  calibrated.request(4,Stage::Policy,now); assert(!calibrated.moving); // No repeated addition.
  calibrated.request(3,Stage::Policy,now);
  calibrated.step(now+3000000000,false); assert(calibrated.q==compensated_zero);
  now+=3000000000;
  calibrated.request(2,Stage::Policy,now);
  calibrated.step(now+3000000000,false); assert(calibrated.q==grasp);
  now+=3000000000;
  calibrated.request(1,Stage::Policy,now);
  calibrated.step(now+3000000000,false); assert(calibrated.q==zeros);
  now+=3000000000;
  calibrated.request(4,Stage::Policy,now);
  calibrated.step(now+1000000000,false);
  previous=calibrated.q;
  calibrated.step(now+1500000000,true);
  calibrated.request(4,Stage::Policy,now+2000000000);
  calibrated.step(now+3000000000,false);
  assert(calibrated.q==previous && !calibrated.active && calibrated.dq==zeros);
  HandCore uncalibrated(true,grasp);
  feedback.received_ns.fill(now);
  assert(uncalibrated.prepare(feedback,now));
  uncalibrated.step(now+3000000000,false);
  uncalibrated.request(3,Stage::Holding,now+3000000000);
  uncalibrated.request(4,Stage::Holding,now+3000000000);
  assert(!uncalibrated.moving && uncalibrated.pose==1);
  // A valid offset must not allow a grasp + offset target beyond motor limits.
  auto unsafe_sum=compensated_zero; unsafe_sum[10]=1.f;
  HandCore limited(true,grasp,unsafe_sum);
  assert(limited.prepare(feedback,now));
  limited.step(now+3000000000,false);
  limited.request(4,Stage::Holding,now+3000000000);
  assert(!limited.moving && limited.pose==1);
  limited.request(3,Stage::Holding,now+3000000000);
  assert(limited.moving && limited.pose==3);
  rejected=false;
  try { HandCore invalid(true,grasp,bad); } catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
  // Firmware that publishes only rt/lf/dex3/*/state must unlock preparation.
  HandFeedbackSources sources;
  assert(sources.snapshot(now).healthy_flags(now)==0);
  assert(sources.topic(0,0)=="rt/lf/dex3/left/state");
  assert(sources.topic(1,0)=="rt/lf/dex3/right/state");
  auto& primary=sources.samples[0];
  primary.valid.fill(true); primary.received_ns.fill(now); primary.q=grasp;
  auto selected=sources.snapshot(now);
  assert(selected.healthy_flags(now)==3);
  HandCore field_hands(true,grasp);
  assert(field_hands.prepare(selected,now));
  assert(field_hands.q==grasp); // Real initial angles, never a fabricated all-zero start.
  // Legacy SDK topic remains usable, including different firmware per hand.
  assert(sources.topic(0,1)=="rt/dex3/left/state");
  assert(sources.topic(1,1)=="rt/dex3/right/state");
  primary.received_ns[1]=0;
  auto& alternate=sources.samples[1];
  alternate.valid.fill(true); alternate.received_ns.fill(now); alternate.q=grasp;
  selected=sources.snapshot(now);
  assert(selected.source[0]==0 && selected.source[1]==1 && selected.healthy_flags(now)==3);
  // Fresh but faulty primary feedback cannot be hidden by healthy legacy feedback.
  primary.valid[0]=false; primary.invalid_reason[0]="motor_error index=1 code=7";
  selected=sources.snapshot(now);
  assert(selected.source[0]==0 && selected.healthy_flags(now)==2);
  assert(selected.problem(0,now)==primary.invalid_reason[0]);
  assert(!field_hands.prepare(selected,now));
  // Once primary stops arriving, switch to the fresh alternate; neither topic may
  // refresh the other hand, and stale data never unlocks the preparation gate.
  alternate.received_ns.fill(now+210000000);
  selected=sources.snapshot(now+210000000);
  assert(selected.source[0]==1 && selected.healthy_flags(now+210000000)==3);
  selected=sources.snapshot(now+420000000);
  assert(selected.healthy_flags(now+420000000)==0);
  assert(selected.problem(0,now+420000000)=="stale_over_200ms");
  assert(!field_hands.prepare(selected,now+420000000));
  assert(HandFeedback{}.problem(0,now)=="no_sample");
  // Live editing cannot enable the hand, interrupt the first ramp, or resume a fault.
  HandCore editor(true,grasp,compensated_zero);
  feedback.valid.fill(true); feedback.received_ns.fill(now); feedback.q=zeros;
  CommandPacket edit; edit.hand_pose=2; edit.hand_grasp_mode=2;
  std::copy(grasp.begin(),grasp.end(),edit.hand_grasp);
  assert(!editor.command(edit,Stage::Holding,now,feedback));
  assert(!editor.ever_enabled);
  editor.request(0,Stage::Holding,now,&feedback);
  editor.request(2,Stage::Holding,now,&feedback);
  assert(!editor.command(edit,Stage::Holding,now,feedback));
  now+=3000000000; editor.step(now,false); feedback.received_ns.fill(now);
  edit.hand_grasp[0]=.2f;
  assert(editor.command(edit,Stage::Holding,now,feedback));
  assert(editor.editing);
  previous=editor.q;
  edit.hand_grasp[7]=2.f;
  assert(!editor.command(edit,Stage::Holding,now,feedback));
  assert(editor.q==previous);
  edit.hand_grasp[7]=.5f; edit.hand_pose=4; // Valid raw, invalid compensated.
  assert(!editor.command(edit,Stage::Holding,now,feedback));
  edit.hand_grasp[7]=.1f;
  assert(editor.command(edit,Stage::Holding,now,feedback));
  for(int tick=0;tick<600;++tick) {
    now+=5000000; feedback.received_ns.fill(now);
    assert(editor.command(edit,Stage::Holding,now,feedback));
    editor.step(now,false);
    for(int i=0;i<14;++i) assert(std::abs(editor.q[i]-previous[i])<=.002501f);
    previous=editor.q;
  }
  assert(!editor.moving && std::abs(editor.q[7]-.8f)<1e-6);
  feedback.valid[0]=false;
  assert(editor.check_feedback(feedback,now));
  assert(!editor.active && !editor.editing);
  feedback.valid[0]=true;
  editor.request(0,Stage::Holding,now,&feedback);
  assert(!editor.command(edit,Stage::Holding,now,feedback));
  assert(!editor.active); // Even after idle acknowledgement, sliders cannot enable.
  std::cout<<"Dex3 primary/legacy state topics, mixed sources and health gates passed\n";
  std::cout<<"Dex3 prepare, 3s open/grasp, smooth endpoints, busy rejection and stop passed\n";
  std::cout<<"Compensated zero/grasp, no accumulation, target limits, original targets and stop passed\n";
}
