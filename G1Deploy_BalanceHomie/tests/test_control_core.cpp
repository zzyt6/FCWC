#include "control_core.hpp"
#include <cassert>
#include <iostream>
#include <limits>

using namespace decoupled;
struct Fixture {
  ControlCore core{42}; Feedback f; uint64_t now=1000000000ULL,sequence=0;
  Fixture() {
    f.valid=true; f.remote_valid=true; f.packet.received_ns=now; f.packet.sequence=1;
    f.packet.quaternion[0]=1;
    for(int i=0;i<29;++i) f.packet.q[i]=config::default_q_sdk29[i];
  }
  void advance(double seconds=0.005) {
    now+=uint64_t(seconds*1e9); f.packet.received_ns=now; ++f.packet.sequence;
  }
  CommandPacket packet(Request request) {
    CommandPacket c; c.session=42; c.sequence=++sequence; c.config_token=config::token;
    c.observed_sequence=f.packet.sequence; c.observed_received_ns=f.packet.received_ns;
    c.request=static_cast<uint32_t>(request);
    for(int i=0;i<29;++i) c.q[i]=config::default_q_sdk29[i];
    return c;
  }
  void send(Request r) { assert(core.command(packet(r),f,now)); }
  void holding() {
    send(Request::Prepare); core.step(f,now);
    for(int i=0;i<620;++i) {
      advance(); send(Request::Prepare); auto out=core.step(f,now);
      for(int j=0;j<29;++j) f.packet.q[j]=out.q[j];
    }
    assert(core.stage==Stage::Holding);
  }
  CommandPacket upper(float target,bool active=true) {
    auto c=packet(Request::Policy); c.upper_mode=1; c.upper_target_active=active;
    for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) c.q[i]=target;
    return c;
  }
  void pushing() {
    holding(); send(Request::Policy); core.step(f,now);
    for(int tick=0;tick<105;++tick) {
      advance(); assert(core.command(upper(0),f,now)); auto out=core.step(f,now);
      for(int i=0;i<29;++i) f.packet.q[i]=out.q[i];
    }
    assert(core.step(f,now).upper_gain_blend==1);
  }
};
int main() {
  { Fixture x; for(int i=0;i<100;++i) { x.advance(); assert(!x.core.step(x.f,x.now).publish); }
    x.send(Request::Policy); assert(!x.core.ever_enabled); }
  { Fixture x; auto c=x.packet(Request::Prepare); c.session=99;
    assert(!x.core.command(c,x.f,x.now)); assert(!x.core.ever_enabled); }
  { Fixture x; auto c=x.packet(Request::Prepare); c.config_token++;
    assert(!x.core.command(c,x.f,x.now)); assert(!x.core.ever_enabled); }
  { Fixture x; auto c=x.packet(Request::Prepare); c.q[0]=std::numeric_limits<float>::quiet_NaN();
    assert(!x.core.command(c,x.f,x.now)); assert(!x.core.ever_enabled); }
  { Fixture x; x.f.packet.received_ns-=200000000;
    assert(!x.core.command(x.packet(Request::Prepare),x.f,x.now)); assert(!x.core.ever_enabled); }
  { Fixture x; x.f.packet.q[0]=0.4; x.f.packet.q[15]=0.25; x.f.packet.q[13]=0.1; x.holding();
    auto out=x.core.step(x.f,x.now);
    for(int i=0;i<12;++i) assert(std::abs(out.q[i]-config::default_q_sdk29[i])<1e-5);
    assert(std::abs(out.q[15]-0.25)<1e-5); assert(std::abs(out.q[13]-.1)<1e-5);
    assert(out.kp[0]==config::balance_kp_sdk29[0] && out.kp[3]==config::balance_kp_sdk29[3]); }
  { Fixture x; x.holding(); auto before=x.core.step(x.f,x.now); x.advance();
    auto c=x.packet(Request::Policy); c.q[0]+=1;
    assert(x.core.command(c,x.f,x.now)); auto after=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy);
    // Direct target still obeys the nominal 88 Nm PD effort bound.
    assert(after.q[0]>before.q[0]+.8);
    assert(std::abs(after.kp[0]*(after.q[0]-x.f.packet.q[0]))<=88.001); }
  { Fixture x; x.holding(); x.advance(0.2); auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault && out.publish && out.kp[0]==0 && out.kd[0]==2);
    assert(!x.core.command(x.packet(Request::Prepare),x.f,x.now)); }
  { Fixture x; x.holding(); x.now+=200000000; auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault && out.kp[0]==0); }
  { Fixture x; x.holding(); x.f.packet.buttons=kL2; auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Damped && out.kp[0]==0); }
  { Fixture x; x.holding(); x.f.packet.rpy[0]=1.; x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault); }
  { Fixture x; x.holding(); x.f.motor_error=true; x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault); }
  { Fixture x; x.holding();
    x.f.motor_temperature[19]={45,int16_t(config::max_temperature_c)};
    std::string reason;
    assert(x.core.healthy(x.f,x.now,&reason) && reason.empty());
    ++x.f.motor_temperature[19][1];
    x.f.motor_temperature[26]={int16_t(config::max_temperature_c+2),46};
    assert(!x.core.healthy(x.f,x.now));
    assert(!x.core.healthy(x.f,x.now,&reason));
    assert(reason=="motor_temperature_high limit_c="+std::to_string(config::max_temperature_c)
                  +" index=19 c="+std::to_string(int(config::max_temperature_c+1))
                  +" temperatures=45,"+std::to_string(int(config::max_temperature_c+1))
                  +" index=26 c="+std::to_string(int(config::max_temperature_c+2))
                  +" temperatures="+std::to_string(int(config::max_temperature_c+2))+",46");
    auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault && out.kp[0]==0 && out.kd[0]==config::damping_kd);
    assert(x.core.reason=="feedback unhealthy: "+reason);
    x.f.motor_temperature={};
    assert(x.core.healthy(x.f,x.now));
    x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault && x.core.reason=="feedback unhealthy: "+reason); }
  // Every SDK joint and either temperature channel must identify the source.
  for(int joint=0;joint<29;++joint) for(int sensor=0;sensor<2;++sensor) {
    Fixture x; x.f.motor_temperature[joint][sensor]=int16_t(config::max_temperature_c+1);
    std::string reason;
    assert(!x.core.healthy(x.f,x.now,&reason));
    assert(reason.find(" index="+std::to_string(joint)+" c=")!=std::string::npos);
    ControlCore single_waist(42,false,true,true);
    assert(single_waist.healthy(x.f,x.now)==(joint==13 || joint==14));
  }
  { Fixture x; x.holding(); x.f.motor_temperature[0][0]=int16_t(config::max_temperature_c+1);
    assert(!x.core.command(x.packet(Request::Policy),x.f,x.now));
    assert(x.core.stage==Stage::Fault);
    assert(x.core.reason.find("command rejected: motor_temperature_high") == 0);
    assert(x.core.reason.find(" index=0 c=")!=std::string::npos); }
  { Fixture x; x.holding(); x.f.remote_valid=false; x.core.step(x.f,x.now);
    assert(x.core.stage==(config::require_remote_heartbeat?Stage::Fault:Stage::Holding)); }
  { Fixture x; x.holding(); auto c=x.packet(Request::Hold); assert(x.core.command(c,x.f,x.now));
    assert(!x.core.command(c,x.f,x.now)); }
  { Fixture x; x.holding(); auto c=x.packet(Request::Policy); c.q[0]=5;
    assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy && x.core.target_warning_mask()==1);
    assert(out.kp[0]>0 && out.q[0]<=config::joint_upper_sdk29[0]);
    x.advance(); x.send(Request::Policy); assert(x.core.target_warning_mask()==0); }
  { Fixture x; x.holding(); auto c=x.packet(Request::Policy); c.q[0]=std::numeric_limits<float>::quiet_NaN();
    assert(!x.core.command(c,x.f,x.now)); assert(x.core.stage==Stage::Fault); }
  { Fixture x; x.holding();
    for(int i=0;i<40;++i) {
      x.advance(); auto c=x.packet(Request::Policy); c.q[0]=2.5;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      assert(std::abs(out.kp[0]*(out.q[0]-x.f.packet.q[0])-out.kd[0]*x.f.packet.dq[0])<=88.001);
    } }
  { Fixture x; x.holding(); x.f.packet.dq[0]=500; auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Holding && out.kp[0]>0 && out.kp[0]<config::balance_kp_sdk29[0]);
    assert(x.core.effort_warning_mask()==1);
    assert(std::abs(double(out.kp[0])*(out.q[0]-x.f.packet.q[0])-out.kd[0]*x.f.packet.dq[0])<=88.001);
    for(int i=1;i<29;++i) assert(out.kp[i]==config::balance_kp_sdk29[i]); }
  { Fixture x; x.f.packet.q[0]=0.4; x.send(Request::Prepare); x.core.step(x.f,x.now);
    for(int i=0;i<1000;++i) { x.advance(); x.send(Request::Prepare); x.core.step(x.f,x.now); }
    assert(x.core.stage==Stage::Ramping);
    auto before=x.core.step(x.f,x.now);
    x.advance(5.0); x.send(Request::Prepare); x.core.step(x.f,x.now);
    assert(!x.core.preparation_overdue(x.now));
    // Crossing 10 s and remaining stuck for another minute must never remove
    // position support, change the target/gains or claim measured readiness.
    x.advance(); x.send(Request::Prepare);
    for(int i=0;i<60;++i) {
      auto out=x.core.step(x.f,x.now);
      assert(x.core.stage==Stage::Ramping && x.core.preparation_overdue(x.now));
      assert(x.core.healthy(x.f,x.now) && out.publish);
      assert(out.q==before.q && out.kp==before.kp && out.kd==before.kd);
      x.advance(1.0); x.send(Request::Prepare);
    }
    x.send(Request::Policy); x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Ramping);
    for(int j=0;j<29;++j) x.f.packet.q[j]=before.q[j];
    x.advance(); x.send(Request::Prepare); auto arrived=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Holding && !x.core.preparation_overdue(x.now));
    assert(arrived.kp==before.kp && arrived.kd==before.kd);
    x.advance(); x.send(Request::Policy); x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy); }
  // A preparation warning must not suppress an independent fault or stop.
  for(int trigger=0;trigger<6;++trigger) {
    Fixture x; x.f.packet.q[0]=.4; x.send(Request::Prepare); x.core.step(x.f,x.now);
    x.advance(10.1); x.send(Request::Prepare); x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Ramping && x.core.preparation_overdue(x.now));
    if(trigger==0) x.send(Request::Stop);
    if(trigger==1) x.f.packet.buttons=kL2;
    if(trigger==2) x.f.packet.received_ns-=200000000;
    if(trigger==3) x.advance(.2); // fresh feedback, expired command heartbeat
    if(trigger==4) x.f.motor_error=true;
    if(trigger==5) x.f.packet.rpy[0]=1.;
    auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==(trigger<2?Stage::Damped:Stage::Fault));
    assert(!x.core.preparation_overdue(x.now));
    assert(out.publish && out.kp[0]==0 && out.kd[0]==config::damping_kd);
  }
  { Fixture x; x.holding(); auto before=x.core.step(x.f,x.now); x.advance();
    auto c=x.packet(Request::Policy); c.q[12]=0.2; c.q[13]=0.1; c.q[14]=-0.1; c.q[15]=0.4;
    assert(x.core.command(c,x.f,x.now)); auto after=x.core.step(x.f,x.now);
    for(int i=12;i<29;++i) assert(after.q[i]==before.q[i]);
    assert(after.q[15]==before.q[15]);
    assert(after.kp[14]==config::balance_kp_sdk29[14] && after.kd[14]==config::balance_kd_sdk29[14]); }
  { Fixture x; x.f.remote_valid=false; std::string reason;
    ControlCore required(42,true), ignored(42,false);
    assert(!required.healthy(x.f,x.now,&reason));
    assert(reason=="wirelesscontroller_missing_or_stale");
    assert(ignored.healthy(x.f,x.now));
    x.f.packet.received_ns-=200000000;
    assert(!ignored.healthy(x.f,x.now,&reason));
    assert(reason=="lowstate_missing_or_stale"); }
  { Fixture x; x.core=ControlCore(42,false); x.f.remote_valid=false; x.holding();
    x.f.packet.buttons=kL2; auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Damped && out.kp[0]==0 && out.kd[0]==2); }
  { Fixture x; x.core=ControlCore(42,false); x.f.remote_valid=false; x.holding();
    x.f.packet.received_ns=x.now+1; x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault);
    assert(x.core.reason.find("lowstate_timestamp_in_future")!=std::string::npos); }
  { Fixture x; x.f.packet.q[11]=.2; x.send(Request::Prepare); x.core.step(x.f,x.now);
    auto diagnostic=x.core.calibration_diagnostics(x.f,x.now);
    assert(diagnostic.find("index=11")!=std::string::npos); }
  { Fixture x; x.holding(); x.send(Request::Policy); auto before=x.core.step(x.f,x.now);
    // A 0.48 rad ankle correction must reach the next servo tick, in both
    // directions, without frozen-target blending or a 10 rad/s ramp.
    for(int i=0;i<12;++i) {
      x.advance(); auto c=x.packet(Request::Policy); c.profile=(i+1)%2;
      c.q[4]+=(i%2==0?.48f:0.f); c.q[12]=.4; c.q[15]=.5;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      assert(out.profile==c.profile && out.gain_blend==1);
      assert(std::abs(out.q[4]-c.q[4])<1e-6);
      const auto& kp=c.profile==0?config::balance_kp_sdk29:config::homie_kp_sdk29;
      const auto& kd=c.profile==0?config::balance_kd_sdk29:config::homie_kd_sdk29;
      assert(out.kp==kp && out.kd==kd);
      for(int j=12;j<29;++j) assert(out.q[j]==before.q[j]);
      for(int j=0;j<29;++j) {
        assert(std::abs(out.kp[j]*(out.q[j]-x.f.packet.q[j])-out.kd[j]*x.f.packet.dq[j])<=config::effort_sdk29[j]+.001);
        x.f.packet.q[j]=out.q[j];
      }
    }
    x.advance(.2); auto held=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy && x.core.policy_hold() && held.kp[0]>0);
    x.advance(config::command_hold_timeout_s); auto stopped=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Fault && stopped.kp[0]==0 && stopped.kd[0]==config::damping_kd); }
  { Fixture x; x.holding();auto c=x.packet(Request::Policy);c.profile=99;
    assert(!x.core.command(c,x.f,x.now));assert(x.core.stage==Stage::Fault); }
  { Fixture x; x.holding();auto c=x.packet(Request::Policy);c.profile=1;
    assert(x.core.command(c,x.f,x.now));x.core.step(x.f,x.now);
    x.f.packet.buttons=kL2;auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Damped && out.kp[0]==0 && out.kd[0]==config::damping_kd); }
  { Fixture x; x.core=ControlCore(42,false,true,true);
    auto c=x.packet(Request::Prepare);
    c.q[12]=.000117f; c.q[15]=-.674523f; c.q[20]=1.295205f; c.q[27]=1.294801f;
    assert(x.core.command(c,x.f,x.now)); auto initial=x.core.step(x.f,x.now);
    assert(initial.q[15]==x.f.packet.q[15]);
    for(int i=0;i<800;++i) {
      x.advance(); c.sequence=++x.sequence; c.observed_sequence=x.f.packet.sequence;
      c.observed_received_ns=x.f.packet.received_ns;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      for(int j=0;j<29;++j) if(j!=20) x.f.packet.q[j]=out.q[j];
    }
    assert(x.core.stage==Stage::Ramping); // An arm that has not arrived must block readiness.
    assert(x.core.calibration_diagnostics(x.f,x.now).find("index=20")!=std::string::npos);
    for(int i=0;i<500;++i) {
      x.advance(); c.sequence=++x.sequence; c.observed_sequence=x.f.packet.sequence;
      c.observed_received_ns=x.f.packet.received_ns;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      for(int j=0;j<29;++j) x.f.packet.q[j]=out.q[j];
    }
    x.advance();
    auto ready=x.core.step(x.f,x.now); assert(x.core.stage==Stage::Holding);
    for(int j: {12,15,20,27}) assert(std::abs(ready.q[j]-c.q[j])<1e-5);
    x.advance(); auto run=x.packet(Request::Policy); run.profile=1;
    assert(x.core.command(run,x.f,x.now)); auto policy=x.core.step(x.f,x.now);
    for(int j=12;j<29;++j) assert(policy.q[j]==ready.q[j]);
    x.advance(); run=x.packet(Request::Policy); run.profile=0;
    assert(x.core.command(run,x.f,x.now)); policy=x.core.step(x.f,x.now);
    for(int j=12;j<29;++j) assert(policy.q[j]==ready.q[j]);
  }
  // Every configured joint, both bounds and both policy profiles: crossing a
  // position reference cannot reject the next command or remove body effort.
  for(int profile=0;profile<2;++profile) for(int joint=0;joint<29;++joint) for(int direction:{-1,1}) {
    Fixture x; x.holding();
    for(int tick=0;tick<=100;++tick) {
      x.advance(); auto c=x.packet(Request::Policy); c.profile=profile;
      assert(x.core.command(c,x.f,x.now)); x.core.step(x.f,x.now);
    }
    x.f.packet.q[joint]=direction<0?config::joint_lower_sdk29[joint]-.03f:
                                   config::joint_upper_sdk29[joint]+.03f;
    std::string detail;
    assert(x.core.healthy(x.f,x.now,&detail) && detail.empty());
    assert(x.core.position_warning_mask(x.f)==(1U<<joint));
    x.advance(); auto command=x.packet(Request::Policy); command.profile=profile;
    assert(x.core.command(command,x.f,x.now)); auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy && out.publish);
    for(int i=0;i<29;++i) assert(out.kp[i]>0);
    assert(out.q[joint]>=config::joint_lower_sdk29[joint] && out.q[joint]<=config::joint_upper_sdk29[joint]);
    assert(std::abs(double(out.kp[joint])*(out.q[joint]-x.f.packet.q[joint])-
                    double(out.kd[joint])*x.f.packet.dq[joint])<=config::effort_sdk29[joint]+.001);
    x.f.packet.q[joint]=out.q[joint];
    assert(x.core.position_warning_mask(x.f)==0);
    x.advance(); command=x.packet(Request::Policy); command.profile=profile;
    assert(x.core.command(command,x.f,x.now)); assert(x.core.step(x.f,x.now).kp[0]>0);
    // A larger excursion used to cause the separate effort/range global fault.
    const auto& kp=profile?config::homie_kp_sdk29:config::balance_kp_sdk29;
    x.f.packet.q[joint]=direction<0?config::joint_lower_sdk29[joint]-2*config::effort_sdk29[joint]/kp[joint]:
                                   config::joint_upper_sdk29[joint]+2*config::effort_sdk29[joint]/kp[joint];
    auto previous=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Policy && x.core.effort_warning_mask()==(1U<<joint));
    assert(previous.kp[joint]>0 && previous.kp[joint]<kp[joint]);
    for(int i=0;i<29;++i) if(i!=joint) assert(previous.kp[i]==out.kp[i]);
    assert(std::abs(double(previous.kp[joint])*(previous.q[joint]-x.f.packet.q[joint]))<=config::effort_sdk29[joint]+.001);
    x.f.packet.q[joint]=previous.q[joint];
    x.advance(); command=x.packet(Request::Policy); command.profile=profile;
    assert(x.core.command(command,x.f,x.now)); out=x.core.step(x.f,x.now);
    assert(x.core.effort_warning_mask()==0 && out.kp[joint]==kp[joint]);
  }
  // Out-of-reference feedback cannot block takeover or full-body preparation.
  for(int joint=0;joint<29;++joint) for(int direction:{-1,1}) {
    Fixture x; x.core=ControlCore(42,false,true);
    x.f.packet.q[joint]=direction<0?config::joint_lower_sdk29[joint]-.03f:
                                   config::joint_upper_sdk29[joint]+.03f;
    assert(x.core.healthy(x.f,x.now)); x.send(Request::Prepare);
    for(int tick=0;tick<1800 && x.core.stage!=Stage::Holding;++tick) {
      x.advance(); x.send(Request::Prepare); auto out=x.core.step(x.f,x.now);
      assert(x.core.stage==Stage::Ramping || x.core.stage==Stage::Holding);
      for(int i=0;i<29;++i) {
        assert(out.kp[i]>0);
        x.f.packet.q[i]+=.15f*(out.q[i]-x.f.packet.q[i]);
      }
    }
    assert(x.core.stage==Stage::Holding);
  }
  // Push-cart PD and targets are independent of the leg model. In hold mode
  // arm targets are ignored; in push mode all 15 waist/arm targets reach PD.
  { Fixture x; x.holding();
    auto c=x.packet(Request::Policy); c.q[17]=.03f;
    assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
    assert(out.q[17]==0 && out.upper_mode==0 && out.upper_gain_blend==1);
    auto held=out.q;
    for(int tick=0;tick<=110;++tick) {
      x.advance(); c=x.packet(Request::Policy); c.upper_mode=1; c.upper_target_active=1; c.profile=tick>=30?1:0;
      for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) c.q[i]=.03f;
      assert(x.core.command(c,x.f,x.now)); out=x.core.step(x.f,x.now);
      assert(out.upper_mode==1 && out.kp[0]==100 && out.kd[0]==2);
      if(tick==0) assert(out.upper_gain_blend==0 && out.kp[17]==200 && out.q[17]==held[17]);
      if(out.upper_gain_blend<1) assert(out.q[17]==held[17]);
      for(int i=0;i<29;++i) x.f.packet.q[i]=out.q[i];
    }
    assert(out.upper_gain_blend==1 && out.profile==1);
    for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) {
      assert(out.kp[i]==config::pushcart_upper_kp_sdk29[i]);
      assert(out.kd[i]==config::pushcart_upper_kd_sdk29[i]);
      assert(std::abs(out.q[i]-.03f)<1e-6);
    }
    assert(out.kp[12]==300 && out.kd[12]==3);
    assert(out.kp[15]==100 && out.kd[15]==2 && out.kp[17]==50 && out.kd[17]==2);
    assert(out.kp[18]==50 && out.kd[18]==2 && out.kp[19]==20 && out.kd[19]==1);
    held=out.q;
    for(int tick=0;tick<=110;++tick) {
      x.advance(); c=x.packet(Request::Policy); c.upper_mode=0; c.profile=0;
      c.q[17]=-.2f; // Returning to ordinary mode holds the last executed arm pose.
      assert(x.core.command(c,x.f,x.now)); out=x.core.step(x.f,x.now);
      assert(out.upper_mode==0 && out.q[17]==held[17]);
      for(int i=0;i<29;++i) x.f.packet.q[i]=out.q[i];
    }
    assert(out.upper_gain_blend==1 && out.kp[17]==200 && out.kd[17]==4 && out.kd[19]==.5f);
    x.advance(); c=x.packet(Request::Policy);c.upper_mode=1;
    assert(x.core.command(c,x.f,x.now));x.core.step(x.f,x.now);
    x.send(Request::Stop);out=x.core.step(x.f,x.now);
    assert(x.core.stage==Stage::Damped && out.kp[17]==0 && out.kd[17]==config::damping_kd);
  }
  { Fixture x; auto c=x.packet(Request::Prepare); c.upper_mode=1;
    assert(!x.core.command(c,x.f,x.now)); assert(!x.core.ever_enabled); }

  // Real 50 -> 200 Hz cadence: four evenly spaced upper targets, direct legs.
  // A repeated endpoint is a heartbeat, not a new interpolation segment.
  { Fixture x; x.pushing();
    for(int tick=1;tick<=7;++tick) {
      x.advance();
      if(tick<=2) {
        auto c=x.upper(.06f); c.q[0]+=.06f; c.profile=tick==2?1:0;
        assert(x.core.command(c,x.f,x.now));
      }
      auto out=x.core.step(x.f,x.now);
      for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) {
        assert(std::abs(out.q[i]-.015f*std::min(tick,4))<1e-6);
        assert(out.kp[i]==config::pushcart_upper_kp_sdk29[i]);
        assert(out.kd[i]==config::pushcart_upper_kd_sdk29[i]);
      }
      assert(std::abs(out.q[0]-config::default_q_sdk29[0]-.06f)<1e-6);
      assert(out.q[13]==0 && out.q[14]==0);
    }
  }
  // Timing jitter uses elapsed time; no tick-count assumption or extrapolation.
  { Fixture x; x.pushing(); double elapsed=0;
    for(double dt: {.005,.007,.003,.005,.010}) {
      x.advance(dt); elapsed+=dt;
      if(elapsed==.005) assert(x.core.command(x.upper(.06f),x.f,x.now));
      auto out=x.core.step(x.f,x.now);
      assert(std::abs(out.q[16]-.06*std::min(elapsed/.02,1.))<1e-6);
    }
  }
  // An early reversal anchors to the last limited output, not measured q=0
  // or the previous +.06 endpoint. Packets in one servo tick use the newest.
  { Fixture x; x.pushing();
    x.advance(); assert(x.core.command(x.upper(.06f),x.f,x.now)); x.core.step(x.f,x.now);
    x.advance(); auto out=x.core.step(x.f,x.now); assert(std::abs(out.q[16]-.03f)<1e-6);
    for(int tick=1;tick<=4;++tick) {
      x.advance();
      if(tick==1) {
        assert(x.core.command(x.upper(.09f),x.f,x.now));
        assert(x.core.command(x.upper(-.03f),x.f,x.now));
      }
      out=x.core.step(x.f,x.now);
      assert(std::abs(out.q[16]-(.03f-.015f*tick))<1e-6);
    }
  }
  // Zero/camera pause cancels immediately despite an old Python q snapshot;
  // ordinary mode holds that exact output while restoring its original PD.
  for(bool disable: {false,true}) {
    Fixture x; x.pushing(); x.advance();
    assert(x.core.command(x.upper(.06f),x.f,x.now)); auto moving=x.core.step(x.f,x.now);
    assert(moving.q[16]>.014f && moving.q[16]<.016f);
    for(int tick=0;tick<110;++tick) {
      x.advance(); auto c=x.upper(-.3f,false); c.upper_mode=disable?0:1;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) assert(out.q[i]==moving.q[i]);
      assert(out.kp[16]>0 && out.kd[16]>0 && x.core.stage==Stage::Policy);
    }
    if(!disable) {
      x.advance(); assert(x.core.command(x.upper(.06f),x.f,x.now)); auto out=x.core.step(x.f,x.now);
      assert(std::abs(out.q[16]-(moving.q[16]+.25f*(.06f-moving.q[16])))<1e-6);
    }
  }
  // B freezes the arms in the C++ tick, even before another Python packet.
  { Fixture x; x.pushing(); x.advance();
    assert(x.core.command(x.upper(.06f),x.f,x.now)); auto moving=x.core.step(x.f,x.now);
    x.advance(); x.f.packet.buttons=kB; auto held=x.core.step(x.f,x.now);
    assert(held.q==moving.q && held.kp[16]>0 && x.core.stage==Stage::Policy);
    x.advance(); assert(x.core.command(x.upper(-.06f),x.f,x.now)); held=x.core.step(x.f,x.now);
    assert(held.q==moving.q);
    x.advance(); x.f.packet.buttons=0; held=x.core.step(x.f,x.now);
    assert(held.q==moving.q); // Old segment cannot resume on release.
  }
  // Interpolation cannot delay explicit stop, L2, a lost heartbeat or health faults.
  for(int stop=0;stop<4;++stop) {
    Fixture x; x.pushing(); x.advance();
    assert(x.core.command(x.upper(.06f),x.f,x.now)); x.core.step(x.f,x.now);
    x.advance(stop==2?config::command_hold_timeout_s+.005:.005);
    if(stop==0) x.send(Request::Stop);
    if(stop==1) x.f.packet.buttons=kL2;
    if(stop==3) x.f.motor_error=true;
    auto out=x.core.step(x.f,x.now);
    assert(x.core.stage==(stop<2?Stage::Damped:Stage::Fault));
    for(int i=0;i<29;++i) assert(out.kp[i]==0 && out.kd[i]==config::damping_kd);
    assert(!x.core.command(x.upper(.06f),x.f,x.now));
  }
  // Joint/PD effort limits remain after interpolation, and a retarget uses
  // their actual output as its anchor rather than an unreachable endpoint.
  { Fixture x; x.pushing(); x.advance();
    assert(x.core.command(x.upper(100.f),x.f,x.now)); auto limited=x.core.step(x.f,x.now);
    for(int tick=0;tick<4;++tick) {
      for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) {
        assert(limited.q[i]>=config::joint_lower_sdk29[i] && limited.q[i]<=config::joint_upper_sdk29[i]);
        assert(std::abs(limited.kp[i]*(limited.q[i]-x.f.packet.q[i])-limited.kd[i]*x.f.packet.dq[i])
               <=config::effort_sdk29[i]+.001);
      }
      x.advance(); limited=x.core.step(x.f,x.now);
    }
    const auto anchor=limited.q;
    x.advance(); assert(x.core.command(x.upper(0),x.f,x.now)); auto out=x.core.step(x.f,x.now);
    for(int i=0;i<29;++i) if(is_pushcart_upper_joint(i)) assert(std::abs(out.q[i]-.75f*anchor[i])<1e-6);
  }
  { Fixture x; x.pushing(); auto c=x.upper(0); c.upper_target_active=2;
    assert(!x.core.command(c,x.f,x.now)); assert(x.core.stage==Stage::Fault); }
  // A late policy tick holds the last leg targets with Balance gains instead
  // of damping, records the hold, and resumes on the next command.
  { Fixture x; x.holding(); x.send(Request::Policy); x.core.step(x.f,x.now);
    auto c=x.packet(Request::Policy); c.profile=1; c.q[3]+=.1f;
    for(int tick=0;tick<200;++tick) {          // settle into Homie
      x.advance(); c=x.packet(Request::Policy); c.profile=1; c.q[3]+=.1f;
      assert(x.core.command(c,x.f,x.now)); auto out=x.core.step(x.f,x.now);
      for(int j=0;j<29;++j) x.f.packet.q[j]=out.q[j];
    }
    auto last=x.core.step(x.f,x.now);
    assert(last.profile==1 && !x.core.policy_hold());
    // 120 ms without a command, but fresh healthy feedback: hold, not damp.
    for(int tick=0;tick<24;++tick) { x.advance(); last=x.core.step(x.f,x.now); }
    assert(x.core.stage==Stage::Policy && x.core.policy_hold());
    assert(x.core.policy_hold_count()==1 && x.core.policy_hold_gap_ns()>uint64_t(config::command_timeout_s*1e9));
    for(int tick=0;tick<200;++tick) { x.advance(); last=x.core.step(x.f,x.now); }  // +1 s
    assert(x.core.stage==Stage::Policy && x.core.policy_hold());
    assert(last.profile==0 && last.gain_blend==1);   // standing Balance gains
    assert(last.kp==config::balance_kp_sdk29 && last.kd==config::balance_kd_sdk29);
    for(int i=0;i<12;++i) assert(std::abs(last.q[i]-(config::default_q_sdk29[i]+(i==3?.1f:0.f)))<1e-4);
    // The next command resumes the policy and records the hold duration.
    x.advance(); c=x.packet(Request::Policy); c.profile=1; c.q[3]+=.1f;
    assert(x.core.command(c,x.f,x.now)); last=x.core.step(x.f,x.now);
    assert(!x.core.policy_hold() && x.core.stage==Stage::Policy && last.profile==1);
    assert(x.core.last_policy_hold_ns()>uint64_t(1e9));
    // Hard faults are never masked by a hold.
    for(int trigger=0;trigger<5;++trigger) {
      Fixture y; y.holding(); y.send(Request::Policy); y.core.step(y.f,y.now);
      y.advance(.15); y.core.step(y.f,y.now); assert(y.core.policy_hold());
      if(trigger==0) y.f.packet.buttons=kL2;
      if(trigger==1) y.f.motor_error=true;
      if(trigger==2) y.f.packet.rpy[1]=1.;
      if(trigger==3) y.f.motor_temperature[3]={int16_t(config::max_temperature_c+5),40};
      if(trigger==4) y.f.packet.received_ns-=200000000;
      auto out=y.core.step(y.f,y.now);
      assert(y.core.stage==(trigger==0?Stage::Damped:Stage::Fault) && out.kp[0]==0);
    }
  }
  std::cout<<"Control-core preparation, position warnings, local effort limits and push-cart upper PD passed\n";
}
