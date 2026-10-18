#pragma once
#include "hand_core.hpp"

namespace decoupled {
// A truncated DDS sample is not a new measurement. Retain the last complete
// sample and its original age; repeated malformed samples expire the 200 ms
// watchdog. Explicit hardware errors must still invalidate feedback immediately.
template<class Message>
bool receive_hand_feedback(const Message& msg, int side, uint64_t now,
                           float max_temperature, HandFeedback& feedback) {
  std::string error;
  for(auto code:msg.error()) if(code && error.empty()) error="hand_error="+std::to_string(code);
  for(size_t j=0;j<msg.motor_state().size();++j) {
    const auto& motor=msg.motor_state()[j];
    if(error.empty() && motor.motorstate())
      error="motor_error index="+std::to_string(j)+" code="+std::to_string(motor.motorstate());
    for(auto t:motor.temperature()) if(t>=max_temperature && error.empty())
      error="motor_temperature index="+std::to_string(j)+" c="+std::to_string(t);
  }
  if(msg.motor_state().size()!=7 && error.empty()) return false;
  for(size_t j=0;j<std::min(size_t(7),msg.motor_state().size());++j) {
    const auto& motor=msg.motor_state()[j];
    const bool finite=std::isfinite(motor.q()) && std::isfinite(motor.dq());
    if(error.empty() && !finite) error="nonfinite_joint="+std::to_string(j);
    if(finite) { feedback.q[side*7+j]=motor.q(); feedback.dq[side*7+j]=motor.dq(); }
  }
  feedback.valid[side]=error.empty();
  feedback.invalid_reason[side]=error;
  feedback.received_ns[side]=now;
  return true;
}
}
