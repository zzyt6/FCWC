#pragma once
#include "protocol.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace decoupled {
// SDK motor order per hand: thumb 0/1/2, middle 0/1, index 0/1.
constexpr std::array<float,14> hand_lower={-1.05f,-.724f,0,-1.57f,-1.75f,-1.57f,-1.75f,
                                         -1.05f,-1.05f,-1.75f,0,0,0,0};
constexpr std::array<float,14> hand_upper={1.05f,1.05f,1.75f,0,0,0,0,
                                         1.05f,.742f,0,1.57f,1.75f,1.57f,1.75f};
// Recovery is only for a small software-limit excursion, never an SDK fault.
constexpr float hand_recovery_max_excursion=.15f, hand_recovery_inset=.03f;
constexpr float hand_recovery_kp_min=.5f, hand_recovery_kp_max=2.f;
constexpr double hand_recovery_timeout_s=3.;
struct HandFeedback {
  std::array<float,14> q{},dq{};
  std::array<HandTelemetry,2> motor_telemetry{};
  std::array<uint64_t,2> received_ns{};
  std::array<bool,2> valid{};
  std::array<std::string,2> invalid_reason{};
  std::array<int,2> source{{-1,-1}};
  bool fresh(int side,uint64_t now) const {
    return received_ns[side] && now>=received_ns[side] && (now-received_ns[side])*1e-9<=.2;
  }
  std::string problem(int side,uint64_t now) const {
    if(!received_ns[side]) return "no_sample";
    if(now<received_ns[side]) return "timestamp_in_future";
    if(!fresh(side,now)) return "stale_over_200ms";
    if(!valid[side]) return invalid_reason[side].empty()?"invalid_state":invalid_reason[side];
    for(int j=side*7;j<side*7+7;++j) {
      if(!std::isfinite(q[j]) || !std::isfinite(dq[j])) return "nonfinite_joint="+std::to_string(j%7);
      if(q[j]<hand_lower[j]-.02f || q[j]>hand_upper[j]+.02f)
        return "joint_out_of_bounds index="+std::to_string(j%7)+" q="+std::to_string(q[j]);
    }
    return "";
  }
  uint32_t healthy_flags(uint64_t now) const {
    uint32_t flags=0;
    for(int side=0;side<2;++side) {
      bool ok=valid[side] && fresh(side,now);
      for(int j=side*7;j<side*7+7;++j)
        ok=ok && std::isfinite(q[j]) && std::isfinite(dq[j]);
      if(ok) flags|=1U<<side;
    }
    return flags;
  }
  // Position excursions are warnings, independent of SDK/communication health.
  uint32_t position_warning_mask() const {
    uint32_t mask=0;
    for(int j=0;j<14;++j)
      if(std::isfinite(q[j]) && (q[j]<hand_lower[j]-.02f || q[j]>hand_upper[j]+.02f))
        mask|=1U<<j;
    return mask;
  }
  uint32_t recovery_mask(uint64_t now) const {
    uint32_t mask=0;
    for(int side=0;side<2;++side) {
      if(!valid[side] || !fresh(side,now)) continue;
      bool finite=true;
      for(int j=side*7;j<side*7+7;++j) finite=finite && std::isfinite(q[j]) && std::isfinite(dq[j]);
      if(!finite) continue;
      for(int j=side*7;j<side*7+7;++j) {
        float excursion=std::max(hand_lower[j]-q[j],q[j]-hand_upper[j]);
        if(excursion>.02f && excursion<=hand_recovery_max_excursion && std::abs(dq[j])<=.5f)
          mask|=1U<<j;
      }
    }
    return mask;
  }
};

// Firmware variants publish either rt/lf/dex3/*/state or rt/dex3/*/state.
// Keep independent samples: the alternate topic must not refresh stale primary data
// or hide an error in a fresh primary sample. Caller holds the DDS callback mutex.
class HandFeedbackSources {
 public:
  std::array<HandFeedback,2> samples{};
  static std::string topic(int side,int source) {
    return std::string(source==0?"rt/lf/dex3/":"rt/dex3/")+(side==0?"left/state":"right/state");
  }
  HandFeedback snapshot(uint64_t now) const {
    HandFeedback out;
    for(int side=0;side<2;++side) {
      int source=0;
      if(!samples[0].fresh(side,now) && (samples[1].fresh(side,now) ||
          samples[1].received_ns[side]>samples[0].received_ns[side])) source=1;
      const auto& sample=samples[source];
      out.received_ns[side]=sample.received_ns[side];
      out.valid[side]=sample.valid[side];
      out.invalid_reason[side]=sample.invalid_reason[side];
      out.source[side]=sample.received_ns[side]?source:-1;
      out.motor_telemetry[side]=sample.motor_telemetry[side];
      for(int j=side*7;j<side*7+7;++j) { out.q[j]=sample.q[j]; out.dq[j]=sample.dq[j]; }
    }
    return out;
  }
};
class HandCore {
 public:
  static constexpr double duration_s=3.;
  static constexpr float live_speed_rad_s=.5f;
  bool enabled=false,active=false,moving=false,ever_enabled=false,editing=false;
  uint32_t pose=1;
  float progress=0;
  std::array<float,14> q{},dq{};
  bool recovering=false;
  int recovery_joint=-1;
  float recovery_kp=0;
  explicit HandCore(bool use,const std::array<float,14>& grasp,
                    const std::optional<std::array<float,14>>& compensated_zero=std::nullopt)
      :enabled(use),grasp_(grasp),compensated_zero_(compensated_zero) {
    for(int i=0;i<14;++i)
      if(!std::isfinite(grasp[i]) || grasp[i]<hand_lower[i] || grasp[i]>hand_upper[i])
        throw std::runtime_error("Dex3 grasp outside joint limits");
    if(compensated_zero_) for(int i=0;i<14;++i)
      if(!std::isfinite((*compensated_zero_)[i]) || (*compensated_zero_)[i]<hand_lower[i] ||
         (*compensated_zero_)[i]>hand_upper[i])
        throw std::runtime_error("Dex3 compensated zero outside joint limits");
    if(compensated_zero_) {
      std::array<float,14> target{};
      bool valid=true;
      for(int i=0;i<14;++i) {
        target[i]=grasp_[i]+(*compensated_zero_)[i];
        valid=valid && std::isfinite(target[i]) && target[i]>=hand_lower[i] && target[i]<=hand_upper[i];
      }
      if(valid) compensated_grasp_=target;
    }
  }
  bool can_prepare(const HandFeedback& feedback,uint64_t now,uint32_t requested=1) const {
    return enabled && !recovering && feedback.healthy_flags(now)==3 &&
           (requested==0 || requested==1 || (requested==3 && compensated_zero_));
  }
  bool prepare(const HandFeedback& feedback,uint64_t now,uint32_t requested=1) {
    if(!can_prepare(feedback,now,requested)) return false;
    restart_ready_=false;
    q=feedback.q; dq.fill(0); active=ever_enabled=true;
    begin(requested==3?3:1,now); return true;
  }
  void suspend() {
    active=moving=recovering=editing=false; recovery_joint=-1; recovery_kp=0; dq.fill(0);
    restart_ready_=false;
  }
  bool check_feedback(const HandFeedback& feedback,uint64_t now) {
    if(!active || feedback.healthy_flags(now)==3) return false;
    suspend(); return true;
  }
  bool request_recovery(int joint,Stage stage,const HandFeedback& feedback,uint64_t now) {
    if(joint<0) { recovery_latched_=-1; return false; }
    if(recovering) {
      if(joint==recovery_joint) recovery_command_ns_=now;
      return false;
    }
    if(joint>=14 || joint==recovery_latched_ || !enabled || active || moving || stage!=Stage::Waiting ||
       !(feedback.recovery_mask(now)&(1U<<joint))) return false;
    recovery_joint=recovery_latched_=joint;
    recovery_initial_=feedback.q[joint];
    q=feedback.q; dq.fill(0);
    for(int i=0;i<14;++i) q[i]=std::isfinite(q[i])?std::clamp(q[i],hand_lower[i],hand_upper[i]):0.f;
    q[joint]=std::clamp(feedback.q[joint],hand_lower[joint]+hand_recovery_inset,
                       hand_upper[joint]-hand_recovery_inset);
    recovery_start_ns_=recovery_command_ns_=now; recovery_settled_ns_=0;
    recovering=ever_enabled=true; recovery_kp=hand_recovery_kp_min; progress=0;
    return true;
  }
  std::string cancel_recovery(const std::string& reason) {
    // Keep the command latch: a held request must never restart a failed attempt.
    recovering=false; recovery_joint=-1; recovery_kp=0; dq.fill(0);
    return reason;
  }
  std::string step_recovery(const HandFeedback& feedback,uint64_t now) {
    if(!recovering) return "";
    const int j=recovery_joint, side=j/7;
    if(!feedback.valid[side] || !feedback.fresh(side,now)) return cancel_recovery("feedback invalid or stale");
    for(int i=side*7;i<side*7+7;++i)
      if(!std::isfinite(feedback.q[i]) || !std::isfinite(feedback.dq[i])) return cancel_recovery("nonfinite feedback");
    if(now<recovery_command_ns_ || (now-recovery_command_ns_)*1e-9>.25)
      return cancel_recovery("command heartbeat timeout");
    if(now<recovery_start_ns_ || (now-recovery_start_ns_)*1e-9>=hand_recovery_timeout_s)
      return cancel_recovery("measured position did not recover in 3 s");
    float initial_error=std::abs(q[j]-recovery_initial_), error=std::abs(q[j]-feedback.q[j]);
    if(error>initial_error+.02f || feedback.q[j]<hand_lower[j]-hand_recovery_max_excursion ||
       feedback.q[j]>hand_upper[j]+hand_recovery_max_excursion)
      return cancel_recovery("position moved away from recovery target");
    if(std::abs(feedback.dq[j])>.5f) return cancel_recovery("recovery velocity too high");
    // Ramp from 0.5 to 2 over 1 s; retain the feedback/direction/velocity guards.
    recovery_kp=hand_recovery_kp_min+(hand_recovery_kp_max-hand_recovery_kp_min)*
                std::clamp((now-recovery_start_ns_)*1e-9,0.,1.);
    progress=std::clamp(1.f-error/initial_error,0.f,1.f);
    if(error<=.01f && std::abs(feedback.dq[j])<=.05f) {
      if(!recovery_settled_ns_) recovery_settled_ns_=feedback.received_ns[side];
      if(feedback.received_ns[side]>=recovery_settled_ns_+150000000) {
        recovering=false; recovery_joint=-1; recovery_kp=0; progress=1;
      }
    } else recovery_settled_ns_=0;
    return "";
  }
  void request(uint32_t requested,Stage stage,uint64_t now,const HandFeedback* feedback=nullptr) {
    // A fault consumes the current request. Require an idle acknowledgement and
    // a new explicit request before re-enabling; fresh feedback alone is not enough.
    if(requested==0) { if(!active && !recovering) restart_ready_=true; return; }
    if(!enabled || recovering || moving || requested>4 ||
       (requested==3 && !compensated_zero_) ||
       (requested==4 && !compensated_grasp_) ||
       (stage!=Stage::Holding && stage!=Stage::Policy)) return;
    if(!active) {
      if(!restart_ready_ || !feedback || feedback->healthy_flags(now)!=3) return;
      q=feedback->q; dq.fill(0); active=ever_enabled=true; restart_ready_=false;
    } else if(requested==pose) return;
    begin(requested,now);
  }
  // Called only after the body core accepts the session, sequence and heartbeat.
  // Invalid editor input is local to the hand; it must not fault body support.
  bool command(const CommandPacket& cmd,Stage stage,uint64_t now,const HandFeedback& feedback) {
    if(cmd.request==uint32_t(Request::Prepare)) { request(0,stage,now,&feedback); return true; }
    if(cmd.hand_grasp_mode>2) return false;
    if(cmd.hand_grasp_mode) {
      std::array<float,14> raw{}, compensated{};
      bool compensation_valid=bool(compensated_zero_);
      for(int i=0;i<14;++i) {
        raw[i]=cmd.hand_grasp[i];
        if(!std::isfinite(raw[i]) || raw[i]<hand_lower[i] || raw[i]>hand_upper[i]) return false;
        if(compensated_zero_) {
          compensated[i]=raw[i]+(*compensated_zero_)[i];
          compensation_valid=compensation_valid && std::isfinite(compensated[i]) &&
            compensated[i]>=hand_lower[i] && compensated[i]<=hand_upper[i];
        }
      }
      if(cmd.hand_pose==4 && !compensation_valid) return false;
      if(cmd.hand_grasp_mode==2) {
        if(!enabled || !active || recovering || (moving && !editing) ||
           (stage!=Stage::Holding && stage!=Stage::Policy) || feedback.healthy_flags(now)!=3 ||
           (cmd.hand_pose!=2 && cmd.hand_pose!=4)) return false;
        const auto& target=cmd.hand_pose==4?compensated:raw;
        if(!editing || target!=goal_) {
          live_distance_=0;
          for(int i=0;i<14;++i) live_distance_=std::max(live_distance_,std::abs(target[i]-q[i]));
          progress=0; moving=true;
        }
        if(!editing) live_step_ns_=now;
        editing=true; pose=cmd.hand_pose; goal_=target;
      }
      grasp_=raw;
      compensated_grasp_=compensation_valid?std::optional<std::array<float,14>>(compensated):std::nullopt;
    }
    if(cmd.hand_grasp_mode!=2) request(cmd.hand_pose,stage,now,&feedback);
    return true;
  }
  void step(uint64_t now,bool stopped) {
    if(stopped) { suspend(); return; }
    if(active && editing) {
      const double dt=now>=live_step_ns_?std::min((now-live_step_ns_)*1e-9,.02):0.;
      live_step_ns_=now;
      if(dt<=0) return;
      float remaining=0;
      for(int i=0;i<14;++i) {
        const float error=goal_[i]-q[i];
        float delta=std::clamp(error*float(1-std::exp(-dt/.08)),
                               -live_speed_rad_s*float(dt),live_speed_rad_s*float(dt));
        if(std::abs(error)<1e-5f) delta=error;
        q[i]+=delta; dq[i]=delta/dt;
        remaining=std::max(remaining,std::abs(goal_[i]-q[i]));
      }
      moving=remaining>0;
      progress=live_distance_>0?std::clamp(1-remaining/live_distance_,0.f,1.f):1;
      return;
    }
    if(!active || !moving) return;
    double u=std::clamp((now-start_ns_)*1e-9/duration_s,0.,1.);
    // Quintic smoothstep: zero velocity and acceleration at both endpoints.
    double blend=u*u*u*(10+u*(-15+6*u));
    double rate=30*u*u*(1-u)*(1-u)/duration_s;
    for(int i=0;i<14;++i) {
      q[i]=start_[i]+blend*(goal_[i]-start_[i]);
      dq[i]=rate*(goal_[i]-start_[i]);
    }
    progress=u;
    if(u>=1) { q=goal_; dq.fill(0); moving=false; }
  }
  void telemetry(StatePacket& packet,const HandFeedback& feedback,uint64_t now) const {
    packet.hand_flags=enabled?(32U|feedback.healthy_flags(now)|(active?4U:0U)|
                              (moving?8U:0U)|((pose==2 || pose==4)?16U:0U)|(pose>=3?64U:0U)|
                              (recovering?128U:0U)|(editing?256U:0U)):0;
    packet.hand_recovery_mask=enabled && !active && !recovering?feedback.recovery_mask(now):0;
    packet.hand_recovery_joint=recovering?recovery_joint:-1;
    packet.hand_progress=progress;
    for(int i=0;i<14;++i) {
      packet.hand_q[i]=feedback.q[i]; packet.hand_dq[i]=feedback.dq[i];
      packet.hand_target[i]=q[i];
    }
  }
 private:
  bool restart_ready_=false;
  int recovery_latched_=-1;
  float recovery_initial_=0;
  uint64_t recovery_start_ns_=0,recovery_command_ns_=0,recovery_settled_ns_=0;
  void begin(uint32_t requested,uint64_t now) {
    editing=false;
    start_=q; goal_=requested==2?grasp_:std::array<float,14>{};
    if(requested==3) goal_=*compensated_zero_;
    if(requested==4) goal_=*compensated_grasp_;
    pose=requested; start_ns_=now; moving=true; progress=0;
  }
  uint64_t start_ns_=0;
  uint64_t live_step_ns_=0;
  float live_distance_=0;
  std::array<float,14> grasp_{},start_{},goal_{};
  std::optional<std::array<float,14>> compensated_zero_,compensated_grasp_;
};
}
