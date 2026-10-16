#pragma once
#include "protocol.hpp"
#include "generated_config.hpp"
#include <algorithm>
#include <sstream>
#include <string>

namespace decoupled {
class ControlCore {
 public:
  Stage stage=Stage::Waiting;
  std::string reason="waiting for explicit prepare";
  bool ever_enabled=false;
  explicit ControlCore(uint64_t session,bool require_remote_heartbeat=config::require_remote_heartbeat,
                       bool prepare_full_body=false,bool single_waist=false)
      : session_(session),require_remote_heartbeat_(require_remote_heartbeat),
        prepare_full_body_(prepare_full_body),single_waist_(single_waist) {}

  bool healthy(const Feedback& f,uint64_t now,std::string* detail=nullptr) const {
    if(detail) detail->clear();
    auto reject=[detail](const char* reason) {
      if(detail) *detail=reason;
      return false;
    };
    if(!f.valid) return reject("lowstate_missing_or_invalid");
    if(require_remote_heartbeat_ && !f.remote_valid) return reject("wirelesscontroller_missing_or_stale");
    if(f.packet.received_ns>now) return reject("lowstate_timestamp_in_future");
    if(!f.packet.received_ns || (now-f.packet.received_ns)*1e-9>config::state_timeout_s)
      return reject("lowstate_missing_or_stale");
    if(f.motor_error) return reject("motor_status_nonzero");
    bool overheated=false;
    for(int i=0;i<29;++i) {
      if(single_waist_ && (i==13 || i==14)) continue;
      const auto& readings=f.motor_temperature[i];
      const auto temperature=std::max(readings[0],readings[1]);
      if(temperature<=config::max_temperature_c) continue;
      if(!detail) return false;
      if(!overheated) *detail="motor_temperature_high limit_c="+std::to_string(config::max_temperature_c);
      *detail+=" index="+std::to_string(i)+" c="+std::to_string(temperature)
               +" temperatures="+std::to_string(readings[0])+","+std::to_string(readings[1]);
      overheated=true;
    }
    if(overheated) return false;
    double norm=0;
    for(float v:f.packet.quaternion) { if(!std::isfinite(v)) return reject("imu_quaternion_nonfinite"); norm+=v*v; }
    if(norm<0.8 || norm>1.2) return reject("imu_quaternion_norm_invalid");
    for(float v:f.packet.omega) if(!std::isfinite(v)) return reject("imu_angular_velocity_nonfinite");
    for(float v:f.packet.rpy) if(!std::isfinite(v)) return reject("imu_rpy_nonfinite");
    if(std::abs(f.packet.rpy[0])>config::max_tilt_rad ||
       std::abs(f.packet.rpy[1])>config::max_tilt_rad) return reject("body_tilt_high");
    for(int i=0;i<29;++i) {
      if(!std::isfinite(f.packet.q[i]) || !std::isfinite(f.packet.dq[i])) {
        if(detail) *detail="joint_feedback_nonfinite index="+std::to_string(i);
        return false;
      }
    }
    for(float v:f.packet.sticks)
      if(!std::isfinite(v) || std::abs(v)>1.1) return reject("remote_sticks_invalid");
    return true;
  }

  // Model/configuration bounds are a position reference, not a hardware fault.
  uint32_t position_warning_mask(const Feedback& f) const {
    uint32_t mask=0;
    if(!f.valid) return mask;
    for(int i=0;i<29;++i) {
      if(single_waist_ && (i==13 || i==14)) continue;
      if(std::isfinite(f.packet.q[i]) &&
         (f.packet.q[i]<config::joint_lower_sdk29[i]-.02 ||
          f.packet.q[i]>config::joint_upper_sdk29[i]+.02)) mask|=1U<<i;
    }
    return mask;
  }
  uint32_t target_warning_mask() const { return target_warning_mask_; }
  uint32_t effort_warning_mask() const { return effort_warning_mask_; }

  bool preparation_overdue(uint64_t now) const {
    return stage==Stage::Ramping && now>=ramp_start_ &&
           (now-ramp_start_)*1e-9>config::calibration_timeout_s;
  }

  bool command(const CommandPacket& c,const Feedback& f,uint64_t now) {
    if(c.magic!=kCommandMagic || c.version!=kVersion || c.session!=session_ ||
       c.config_token!=config::token || c.sequence<=last_sequence_) return false;
    if(c.request>static_cast<uint32_t>(Request::Stop)) return false;
    last_sequence_=c.sequence;
    auto request=static_cast<Request>(c.request);
    if(request==Request::Stop) { stop("explicit stop",false); return true; }
    if(stage==Stage::Damped || stage==Stage::Fault) return false;
    if(c.profile>static_cast<uint32_t>(Profile::Homie) || c.hand_pose>4 || c.hand_recover < -1 || c.hand_recover>=14 ||
       c.upper_mode>static_cast<uint32_t>(UpperMode::Pushcart) || c.upper_target_active>1 ||
       (c.upper_target_active && c.upper_mode!=static_cast<uint32_t>(UpperMode::Pushcart))) {
      if(ever_enabled) stop("invalid control profile");
      return false;
    }
    auto reject_feedback=[&](const std::string& detail) {
      if(ever_enabled) stop("command rejected: "+detail,true);
      return false;
    };
    if(!healthy(f,now)) {
      std::string detail;
      healthy(f,now,&detail);
      return reject_feedback(detail);
    }
    if(!c.observed_sequence) return reject_feedback("observed_sequence_missing");
    if(c.observed_sequence>f.packet.sequence) return reject_feedback("observed_sequence_ahead_of_bridge");
    if(!c.observed_received_ns) return reject_feedback("observed_timestamp_missing");
    if(c.observed_received_ns>f.packet.received_ns) return reject_feedback("observed_timestamp_ahead_of_bridge");
    if(now<c.observed_received_ns) return reject_feedback("observed_timestamp_in_future");
    if((now-c.observed_received_ns)*1e-9>config::state_timeout_s)
      return reject_feedback("observed_feedback_stale age_ms="+std::to_string((now-c.observed_received_ns)*1e-6));
    uint32_t target_warning=0;
    for(int i=0;i<29;++i) {
      if(!std::isfinite(c.q[i])) {
        if(ever_enabled) stop("invalid target",true);
        return false;
      }
      if(single_waist_ && (i==13 || i==14)) continue;
      if(c.q[i]<config::joint_lower_sdk29[i]-1e-5 || c.q[i]>config::joint_upper_sdk29[i]+1e-5)
        target_warning|=1U<<i;
    }
    target_warning_mask_=target_warning;
    if(request==Request::Wait) return true;
    if(c.upper_mode && (stage!=Stage::Policy || request!=Request::Policy)) return false;
    if(request==Request::Prepare && stage==Stage::Waiting) {
      if(f.packet.buttons&kL2) return false;
      std::copy(std::begin(f.packet.q),std::end(f.packet.q),initial_.begin());
      output_q_=initial_; goal_=initial_;
      // Both policies control only the legs; waist/arms hold the captured pose.
      for(int i=0;i<config::controlled_dofs;++i) goal_[i]=config::default_q_sdk29[i];
      if(prepare_full_body_) std::copy(std::begin(c.q),std::end(c.q),goal_.begin());
      for(int i=0;i<29;++i) goal_[i]=std::clamp(goal_[i],config::joint_lower_sdk29[i],config::joint_upper_sdk29[i]);
      if(single_waist_) for(int i: {13,14}) initial_[i]=goal_[i]=output_q_[i]=0;
      ramp_start_=now; stage=Stage::Ramping; ever_enabled=true; reason="calibrating";
    }
    if(request==Request::Policy && stage==Stage::Holding) {
      stage=Stage::Policy; reason="policy running";
    }
    if(request==Request::Hold && stage==Stage::Policy) {
      stop("policy cannot silently return to hold; restart required",true); return false;
    }
    if(stage==Stage::Policy) {
      if(request!=Request::Policy) { stop("invalid policy transition",true); return false; }
      auto next_profile=static_cast<Profile>(c.profile);
      if(next_profile!=profile_) {
        // Capture the last emitted targets and gains, including interrupted transitions.
        transition_q_=output_q_; start_kp_=active_kp_; start_kd_=active_kd_;
        profile_=next_profile; profile_start_=now;
      }
      auto next_upper=static_cast<UpperMode>(c.upper_mode);
      if(next_upper!=upper_mode_) {
        upper_start_kp_=active_kp_; upper_start_kd_=active_kd_;
        upper_mode_=next_upper; upper_start_=now;
        hold_upper_target();
      }
      for(int i=0;i<config::controlled_dofs;++i)
        goal_[i]=std::clamp(c.q[i],config::joint_lower_sdk29[i],config::joint_upper_sdk29[i]);
      if(upper_mode_==UpperMode::Pushcart && c.upper_target_active && !(f.packet.buttons&kB)) {
        bool changed=!upper_target_active_;
        for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) {
          float target=std::clamp(c.q[i],config::joint_lower_sdk29[i],config::joint_upper_sdk29[i]);
          changed=changed || target!=goal_[i];
          goal_[i]=target;
        }
        if(changed) {
          // A retarget starts at the last emitted (already limited) target,
          // never at the old endpoint or the measured joint position. All
          // packets drained before the next servo tick share this same anchor.
          upper_target_from_=output_q_;
          upper_target_start_=last_step_?last_step_:now;
        }
        upper_target_active_=true;
      } else hold_upper_target();
    }
    command_ns_=now;
    return true;
  }

  void stop(const std::string& why,bool fault=true) {
    if(stage==Stage::Fault || stage==Stage::Damped) return;
    stage=fault?Stage::Fault:Stage::Damped; reason=why;
  }

  std::string calibration_diagnostics(const Feedback& f,uint64_t now) const {
    std::ostringstream out;
    out<<"elapsed_s="<<((now-ramp_start_)*1e-9)
       <<" position_tolerance_rad="<<config::calibration_position_tolerance_rad
       <<" velocity_tolerance_rad_s="<<config::calibration_velocity_tolerance_rad_s
       <<" pending=[";
    bool first=true;
    for(int i=0;i<(prepare_full_body_?29:config::controlled_dofs);++i) {
      if(single_waist_ && (i==13 || i==14)) continue;
      bool target_ok=std::abs(output_q_[i]-goal_[i])<1e-4;
      bool position_ok=std::abs(f.packet.q[i]-goal_[i])<=config::calibration_position_tolerance_rad;
      bool velocity_ok=std::abs(f.packet.dq[i])<=config::calibration_velocity_tolerance_rad_s;
      if(target_ok && position_ok && velocity_ok) continue;
      if(!first) out<<", ";
      first=false;
      out<<"{index="<<i<<" q="<<f.packet.q[i]<<" goal="<<goal_[i]
         <<" commanded_q="<<output_q_[i]<<" dq="<<f.packet.dq[i]
         <<" position_error="<<(f.packet.q[i]-goal_[i])
         <<" target_ok="<<target_ok<<" position_ok="<<position_ok
         <<" velocity_ok="<<velocity_ok<<"}";
    }
    out<<"]";
    return out.str();
  }

  Output step(const Feedback& f,uint64_t now) {
    Output result;
    effort_warning_mask_=0;
    result.profile=static_cast<uint32_t>(profile_);
    result.upper_mode=static_cast<uint32_t>(upper_mode_);
    if(ever_enabled && stage!=Stage::Fault && stage!=Stage::Damped) {
      if(!healthy(f,now)) {
        std::string detail;
        healthy(f,now,&detail);
        stop("feedback unhealthy: "+detail);
      }
      else if(f.packet.buttons&kL2) stop("remote L2",false);
      else if(!command_ns_ || (now-command_ns_)*1e-9>config::command_timeout_s) stop("policy heartbeat timeout");
      // Preparation taking too long is a warning, never a reason to remove
      // position support. Keep the target and gains until measured arrival.
    }
    if(!ever_enabled) return result;
    result.publish=true;
    if(stage==Stage::Fault || stage==Stage::Damped) {
      result.kd.fill(config::damping_kd); return result;
    }
    // Physical B cancels upper motion at 200 Hz, before Python's zero command
    // arrives. Keep position support; L2/explicit stop still select damping.
    if(f.packet.buttons&kB) hold_upper_target();
    double dt=last_step_?std::min((now-last_step_)*1e-9,0.01):1.0/config::servo_hz;
    last_step_=now;
    double elapsed=(now-ramp_start_)*1e-9;
    float transition=(profile_start_ && config::profile_transition_s>0)?
      std::clamp((now-profile_start_)*1e-9/config::profile_transition_s,0.0,1.0):1.0;
    float blend=transition*transition*(3-2*transition);
    result.gain_blend=blend;
    const auto& desired_kp=profile_==Profile::Balance?config::balance_kp_sdk29:config::homie_kp_sdk29;
    const auto& desired_kd=profile_==Profile::Balance?config::balance_kd_sdk29:config::homie_kd_sdk29;
    for(int i=0;i<29;++i) {
      active_kp_[i]=blend<1?start_kp_[i]+blend*(desired_kp[i]-start_kp_[i]):desired_kp[i];
      active_kd_[i]=blend<1?start_kd_[i]+blend*(desired_kd[i]-start_kd_[i]):desired_kd[i];
    }
    // This is an independent waist/arm PD transition; leg-policy switching and
    // the trained Balance/Homie command filter retain their existing timing.
    float upper_t=upper_start_?std::clamp((now-upper_start_)*1e-9/config::pushcart_upper_transition_s,0.0,1.0):1.0;
    float upper_blend=upper_t*upper_t*(3-2*upper_t);
    result.upper_gain_blend=upper_blend;
    if(upper_mode_==UpperMode::Pushcart || upper_blend<1) {
      for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) {
        float kp=upper_mode_==UpperMode::Pushcart?config::pushcart_upper_kp_sdk29[i]:desired_kp[i];
        float kd=upper_mode_==UpperMode::Pushcart?config::pushcart_upper_kd_sdk29[i]:desired_kd[i];
        active_kp_[i]=upper_blend<1?upper_start_kp_[i]+upper_blend*(kp-upper_start_kp_[i]):kp;
        active_kd_[i]=upper_blend<1?upper_start_kd_[i]+upper_blend*(kd-upper_start_kd_[i]):kd;
      }
    }
    if(stage==Stage::Ramping) {
      double u=std::clamp(elapsed/config::calibration_s,0.0,1.0);
      double blend=u*u*(3-2*u);
      for(int i=0;i<29;++i) {
        float desired=initial_[i]+blend*(goal_[i]-initial_[i]);
        float max_delta=config::calibration_speed_rad_s*dt;
        output_q_[i]+=std::clamp(desired-output_q_[i],-max_delta,max_delta);
      }
      bool arrived=elapsed>=config::calibration_s;
      // 目标已到位不代表关节真的到位；实际位置和速度也满足要求才允许开始策略。
      for(int i=0;i<(prepare_full_body_?29:config::controlled_dofs);++i) {
        if(single_waist_ && (i==13 || i==14)) continue;
        arrived=arrived && std::abs(output_q_[i]-goal_[i])<1e-4 &&
        std::abs(f.packet.q[i]-goal_[i])<=config::calibration_position_tolerance_rad &&
        std::abs(f.packet.dq[i])<=config::calibration_velocity_tolerance_rad_s;
      }
      if(arrived) { stage=Stage::Holding; reason="ready; press R2 again for policy"; }
    } else if(stage==Stage::Policy) {
      for(int i=0;i<config::controlled_dofs;++i) {
        float max_delta=config::policy_target_speed_rad_s*dt;
        float target=blend<1?transition_q_[i]+blend*(goal_[i]-transition_q_[i]):goal_[i];
        // A zero rate setting preserves the training policy's direct target.
        // Joint and nominal PD-effort bounds below remain active.
        if(config::policy_target_speed_rad_s>0)
          output_q_[i]+=std::clamp(target-output_q_[i],-max_delta,max_delta);
        else output_q_[i]=target;
      }
      if(upper_mode_==UpperMode::Pushcart && upper_target_active_) {
        if(upper_blend<1) {
          // Do not age an arm trajectory while the independent PD transition
          // holds position. Start interpolation from its final held target.
          upper_target_from_=output_q_;
          upper_target_start_=now;
        } else {
          const double duration=config::pushcart_upper_target_interpolation_s;
          const float alpha=duration>0?
            std::clamp((now-upper_target_start_)*1e-9/duration,0.0,1.0):1.0;
          for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i))
            output_q_[i]=upper_target_from_[i]+alpha*(goal_[i]-upper_target_from_[i]);
        }
      }
    }
    float gain=std::clamp(elapsed/config::gain_ramp_s,0.0,1.0);
    for(int i=0;i<29;++i) {
      result.q[i]=std::clamp(output_q_[i],config::joint_lower_sdk29[i],config::joint_upper_sdk29[i]);
      result.kp[i]=gain*active_kp_[i];
      result.kd[i]=active_kd_[i];
      // 根据当前反馈限制名义 PD 力矩；这是 200 Hz 主机侧估算，不能替代电机内部限流。
      if(result.kp[i]>1e-5) {
        float lo=f.packet.q[i]+(-config::effort_sdk29[i]+result.kd[i]*f.packet.dq[i])/result.kp[i];
        float hi=f.packet.q[i]+( config::effort_sdk29[i]+result.kd[i]*f.packet.dq[i])/result.kp[i];
        lo=std::max(lo,config::joint_lower_sdk29[i]);
        hi=std::min(hi,config::joint_upper_sdk29[i]);
        if(lo>hi) {
          // Outside the reference range, an in-range target may require more
          // nominal PD effort than allowed. Reduce ONLY this joint's gains;
          // preserve its target and all other joints instead of damping the body.
          const double effort=double(result.kp[i])*(double(result.q[i])-f.packet.q[i])-
                              double(result.kd[i])*f.packet.dq[i];
          const double scale=std::min(1.,double(config::effort_sdk29[i])/std::abs(effort));
          result.kp[i]*=scale; result.kd[i]*=scale;
          effort_warning_mask_|=1U<<i;
        } else result.q[i]=std::clamp(result.q[i],lo,hi);
        output_q_[i]=result.q[i];
      }
    }
    return result;
  }
 private:
  void hold_upper_target() {
    upper_target_active_=false;
    for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) goal_[i]=output_q_[i];
  }
  uint32_t target_warning_mask_=0,effort_warning_mask_=0;
  uint64_t session_,last_sequence_=0,command_ns_=0,ramp_start_=0,last_step_=0;
  uint64_t profile_start_=0;
  uint64_t upper_start_=0,upper_target_start_=0;
  Profile profile_=Profile::Balance;
  UpperMode upper_mode_=UpperMode::Hold;
  bool upper_target_active_=false;
  bool require_remote_heartbeat_;
  bool prepare_full_body_,single_waist_;
  std::array<float,29> initial_{},goal_{},output_q_{};
  std::array<float,29> transition_q_{},start_kp_{},start_kd_{};
  std::array<float,29> upper_start_kp_{},upper_start_kd_{};
  std::array<float,29> upper_target_from_{};
  std::array<float,29> active_kp_=config::balance_kp_sdk29,active_kd_=config::balance_kd_sdk29;
};
}
