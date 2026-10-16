#pragma once
#include "hand_core.hpp"

namespace decoupled {
struct HandPd {
  // Standalone bridge callers retain the previous default. Viser supplies the
  // startup JSON values explicitly, including across bridge reacquisitions.
  float kp=.5f, kd=.1f;
  static constexpr float stopped_kd=.1f;
  static HandPd parse(const std::string& text) {
    auto comma=text.find(',');
    if(comma==std::string::npos) throw std::runtime_error("Expected Dex3 kp,kd");
    HandPd gains;
    size_t used=0;
    gains.kp=std::stof(text.substr(0,comma),&used);
    if(used!=comma) throw std::runtime_error("Invalid Dex3 kp");
    gains.kd=std::stof(text.substr(comma+1),&used);
    if(used!=text.size()-comma-1 || !std::isfinite(gains.kp) || gains.kp<=0 ||
       !std::isfinite(gains.kd) || gains.kd<0)
      throw std::runtime_error("Dex3 kp must be finite > 0, kd finite >= 0");
    return gains;
  }
};

template<class Command>
void fill_hand_command(Command& command,const HandCore& hands,int side,const HandPd& gains) {
  command.motor_cmd().resize(7);
  for(int j=0;j<7;++j) {
    auto& motor=command.motor_cmd()[j];
    motor.mode()=static_cast<uint8_t>(j | (1<<4) | (1<<7));
    motor.q()=hands.q[side*7+j]; motor.dq()=hands.active?hands.dq[side*7+j]:0;
    motor.kp()=hands.active?gains.kp:0.f;
    motor.kd()=hands.active?gains.kd:HandPd::stopped_kd;
    motor.tau()=0;
    if(hands.recovering) {
      const bool selected=side*7+j==hands.recovery_joint;
      motor.kp()=selected?hands.recovery_kp:0.f;
      motor.kd()=HandPd::stopped_kd;
      motor.dq()=0;
    }
  }
}

// Hand DDS failures must never unwind through the body publisher/control loop.
// Continue attempting damping writes on both sides after a local failure.
template<class Command,class Publish>
std::array<std::string,2> publish_hand_commands(HandCore& hands,const HandPd& gains,Publish publish) {
  std::array<std::string,2> errors;
  if(!hands.ever_enabled) return errors;
  for(int side=0;side<2;++side) {
    try {
      Command command;
      fill_hand_command(command,hands,side,gains);
      if(!publish(side,command)) errors[side]="DDS publish failed or channel unavailable";
    } catch(const std::exception& e) { errors[side]=e.what(); }
      catch(...) { errors[side]="unknown DDS publish exception"; }
    if(!errors[side].empty() && (hands.active || hands.recovering)) hands.suspend();
  }
  return errors;
}
}
