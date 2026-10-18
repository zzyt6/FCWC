// Socket-free regression against the installed robot SDK's Dex3 wire format.
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include "hand_command.hpp"
#include "hand_feedback.hpp"
#include <org/eclipse/cyclonedds/core/cdr/basic_cdr_ser.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using HandState=unitree_hg::msg::dds_::HandState_;
namespace cdr=org::eclipse::cyclonedds::core::cdr;
using Traits=org::eclipse::cyclonedds::topic::TopicTraits<HandState>;

void require(bool ok,const char* message) {
  if(!ok) throw std::runtime_error(message);
}

// Build XCDR1 independently of the generated serializer, including nine tactile
// sensors. The old bundled IDL omitted two uint32 fields per sensor (72 bytes).
std::vector<uint8_t> fixture() {
  std::vector<uint8_t> data;
  auto integer=[&](uint32_t v,size_t width=4) {
    while(data.size()%width) data.push_back(0);
    for(size_t b=0;b<width;++b) data.push_back(uint8_t(v>>(8*b)));
  };
  auto real=[&](float v) { uint32_t bits; std::memcpy(&bits,&v,4); integer(bits); };
  integer(7);
  for(int j=0;j<7;++j) {
    integer(j,1); real(float(j)/8); real(-float(j)/16); real(0); real(0);
    integer(25,2); integer(26,2); real(24);
    integer(0); integer(0); integer(0);
    for(int i=0;i<4;++i) integer(0);
  }
  integer(9);
  for(int j=0;j<9;++j) {
    for(int i=0;i<12;++i) real(float(j*12+i));
    for(int i=0;i<12;++i) real(float(20+j));
    integer(100+j); integer(200+j); // lost, reserve in the robot's SDK
  }
  for(int i=0;i<13;++i) real(i==0?1.f:0.f); // quaternion, gyro, acceleration, rpy
  integer(28,2);
  real(24); real(.5f); real(5); real(3.3f);
  integer(0); integer(0); integer(0x12345678); integer(0x87654321);
  return data;
}

int main(int argc,char** argv) {
  try {
    // Reproduce the field failure: one empty left-hand message between healthy
    // frames must not drop grasp PD. It must not refresh age or hide a fault.
    {
      using namespace decoupled;
      constexpr uint64_t stamp=1000000000;
      HandFeedback feedback;
      HandState complete;
      complete.motor_state().resize(7);
      require(receive_hand_feedback(complete,0,stamp,80,feedback),"Complete hand sample rejected");
      receive_hand_feedback(complete,1,stamp,80,feedback);
      HandCore hands(true,{});
      require(hands.prepare(feedback,stamp),"Healthy grasp preparation failed");
      HandState empty;
      require(!receive_hand_feedback(empty,0,stamp+5000000,80,feedback),"Empty sample accepted");
      require(feedback.received_ns[0]==stamp,"Empty sample refreshed watchdog");
      require(!hands.check_feedback(feedback,stamp+5000000) && hands.active,"Single empty sample dropped grasp");
      receive_hand_feedback(empty,0,stamp+199000000,80,feedback);
      require(feedback.healthy_flags(stamp+201000000)==0,"Empty stream bypassed watchdog");
      require(hands.check_feedback(feedback,stamp+201000000) && !hands.active,"Stale hand did not suspend");
      receive_hand_feedback(complete,0,stamp+202000000,80,feedback);
      receive_hand_feedback(complete,1,stamp+202000000,80,feedback);
      require(!hands.active,"Recovered feedback rearmed suspended hands");
      empty.error()[0]=7;
      require(receive_hand_feedback(empty,0,stamp+203000000,80,feedback),"Fault in empty sample ignored");
      require(!feedback.valid[0],"Hardware fault was hidden");
      complete.motor_state()[0].motorstate(4);
      receive_hand_feedback(complete,1,stamp+203000000,80,feedback);
      require(!feedback.valid[1],"Motor fault was hidden");
    }
    // Exercise the production command builder with the actual SDK message type.
    auto gains=decoupled::HandPd::parse("1.5,0.2");
    require(gains.kp==1.5f && gains.kd==.2f,"Dex3 PD argument parsing mismatch");
    // No hand motor messages are published until an explicit pose request.
    {
      decoupled::HandCore idle(true,{});
      decoupled::HandFeedback feedback;
      feedback.valid.fill(true); feedback.received_ns.fill(1000000000); feedback.q[0]=.25f;
      int writes=0;
      for(auto stage:{decoupled::Stage::Waiting,decoupled::Stage::Ramping,
                      decoupled::Stage::Holding,decoupled::Stage::Policy}) {
        idle.request(0,stage,1000000000,&feedback);
        idle.step(1000000000,false);
        decoupled::publish_hand_commands<unitree_hg::msg::dds_::HandCmd_>(idle,gains,
          [&](int,const auto&) { ++writes; return true; });
        require(!idle.active && !idle.ever_enabled && writes==0,"Idle body phase published a hand command");
      }
      idle.request(2,decoupled::Stage::Holding,1000000000,&feedback);
      require(idle.active && idle.moving && idle.q==feedback.q,"First pose did not enable from measured angles");
      decoupled::publish_hand_commands<unitree_hg::msg::dds_::HandCmd_>(idle,gains,
        [&](int,const auto& command) {
          ++writes;
          for(const auto& motor:command.motor_cmd()) require(motor.kp()==gains.kp,"First pose did not enable configured PD");
          return true;
        });
      require(writes==2,"First pose did not publish both hands");
    }
    // An SDK write can return false or throw; neither may escape to body control.
    for(int failed_side=0;failed_side<2;++failed_side) for(int failure=0;failure<3;++failure) {
      decoupled::HandCore local(true,{});
      decoupled::HandFeedback healthy;
      healthy.valid.fill(true); healthy.received_ns.fill(1000000000);
      require(local.prepare(healthy,1000000000),"Local prepare failed");
      int writes=0;
      auto errors=decoupled::publish_hand_commands<unitree_hg::msg::dds_::HandCmd_>(local,gains,
        [&](int side,const auto& command)->bool {
          ++writes;
          if(side==failed_side) {
            if(failure==1) throw std::runtime_error("injected SDK exception");
            if(failure==2) throw 7;
            return false;
          }
          if(side>failed_side) for(const auto& motor:command.motor_cmd())
            require(motor.kp()==0 && motor.kd()==.1f,"Failure did not damp remaining hand");
          return true;
        });
      require(writes==2 && !errors[failed_side].empty() && errors[1-failed_side].empty(),"Hand write failure not isolated");
      require(!local.active && !local.moving && !local.recovering,"Faulted hand remains enabled");
      local.request(1,decoupled::Stage::Policy,1000000000,&healthy);
      require(!local.active,"Held request restarted hand after SDK write failure");
      errors=decoupled::publish_hand_commands<unitree_hg::msg::dds_::HandCmd_>(local,gains,
        [&](int,const auto& command) {
          for(const auto& motor:command.motor_cmd())
            require(motor.kp()==0 && motor.kd()==.1f && motor.dq()==0,"Damping retry retained position effort");
          return true;
        });
      require(errors[0].empty() && errors[1].empty() && !local.active,"Transport recovery automatically enabled hand");
      local.request(0,decoupled::Stage::Policy,1000000000,&healthy);
      healthy.q[0]=.3f;
      local.request(1,decoupled::Stage::Policy,1000000000,&healthy);
      require(local.active && local.moving && local.q==healthy.q,"Manual resume must start from current feedback");
    }
    for(const char* invalid:{"nan,0.2","1.5,inf","0,0.2","-1,0.2","1.5,-.2",
                             "1.5","1.5,0.2,0.1","1.5x,0.2","1.5,0.2x","1e40,.2"}) {
      bool rejected=false;
      try { decoupled::HandPd::parse(invalid); } catch(const std::exception&) { rejected=true; }
      require(rejected,"Invalid Dex3 gains accepted");
    }
    decoupled::HandCore hands(true,std::array<float,14>{});
    decoupled::HandFeedback feedback;
    feedback.valid.fill(true); feedback.received_ns.fill(1000000000);
    require(hands.prepare(feedback,1000000000),"Hand prepare failed");
    for(int i=0;i<14;++i) { hands.q[i]=float(i)/100; hands.dq[i]=-float(i)/100; }
    for(const auto& custom:{gains,decoupled::HandPd::parse("1.2,0.15")}) {
      for(int side=0;side<2;++side) {
        unitree_hg::msg::dds_::HandCmd_ command;
        decoupled::fill_hand_command(command,hands,side,custom);
        require(command.motor_cmd().size()==7,"Wrong Dex3 command length");
        for(int j=0;j<7;++j) {
          const auto& motor=command.motor_cmd()[j];
          require(motor.kp()==custom.kp && motor.kd()==custom.kd,"Configured Dex3 gains not sent");
          require(motor.q()==hands.q[side*7+j] && motor.dq()==hands.dq[side*7+j],"Hand trajectory changed");
          require(motor.mode()==uint8_t(j|16|128) && motor.tau()==0,"Hand mode/torque changed");
        }
      }
    }
    hands.step(1000000000,true);
    for(int side=0;side<2;++side) {
      unitree_hg::msg::dds_::HandCmd_ command;
      decoupled::fill_hand_command(command,hands,side,gains);
      for(const auto& motor:command.motor_cmd())
        require(motor.kp()==0 && motor.kd()==.1f && motor.dq()==0 && motor.tau()==0,
                "Configured active PD leaked into stopped commands");
    }
    // Recovery must energize only the selected joint, using bounded gains.
    decoupled::HandCore recovery(true,{});
    feedback.q.fill(0); feedback.q[8]=decoupled::hand_lower[8]-.06f;
    require(recovery.request_recovery(8,decoupled::Stage::Waiting,feedback,1000000000),"Recovery rejected");
    feedback.received_ns.fill(2000000000);
    recovery.request_recovery(8,decoupled::Stage::Waiting,feedback,2000000000);
    require(recovery.step_recovery(feedback,2000000000).empty(),"Recovery failed");
    for(int side=0;side<2;++side) {
      unitree_hg::msg::dds_::HandCmd_ command;
      decoupled::fill_hand_command(command,recovery,side,gains);
      for(int j=0;j<7;++j) {
        const auto& motor=command.motor_cmd()[j];
        require((side*7+j==8)?motor.kp()==2.f:motor.kp()==0,"Unselected recovery motor enabled");
        require(motor.q()>=decoupled::hand_lower[side*7+j] && motor.q()<=decoupled::hand_upper[side*7+j],"Recovery target outside limits");
        require(motor.dq()==0 && motor.tau()==0 && motor.kd()==.1f,"Unsafe recovery command");
      }
    }
    feedback.received_ns.fill(4000000000);
    recovery.request_recovery(8,decoupled::Stage::Waiting,feedback,4000000000);
    require(recovery.step_recovery(feedback,4000000000)=="measured position did not recover in 3 s",
            "Recovery timeout missing");
    for(int side=0;side<2;++side) {
      unitree_hg::msg::dds_::HandCmd_ command;
      decoupled::fill_hand_command(command,recovery,side,gains);
      for(const auto& motor:command.motor_cmd())
        require(motor.kp()==0 && motor.kd()==.1f && motor.dq()==0 && motor.tau()==0,
                "Timed-out recovery retained position effort");
    }
    std::cout<<"Configured Dex3 SDK commands, recovery isolation and fixed stop damping passed\n";
    auto bytes=fixture();
    if(argc==2 && std::string(argv[1])=="--stdin")
      bytes.assign(std::istreambuf_iterator<char>(std::cin),std::istreambuf_iterator<char>());
    require(bytes.size()==1424,"unexpected independent fixture size");
    cdr::basic_cdr_stream stream(cdr::endianness::little_endian);
    stream.set_buffer(bytes.data(),bytes.size());
    HandState state;
    require(cdr::read(stream,state,false),"HandState XCDR1 decode failed");
    require(stream.position()==bytes.size(),"HandState decoder did not consume the entire robot packet");
    require(state.motor_state().size()==7 && state.press_sensor_state().size()==9,"wrong motor/sensor count");
    for(int j=0;j<7;++j) {
      require(state.motor_state()[j].q()==float(j)/8,"motor angle mismatch");
      require(state.motor_state()[j].dq()==-float(j)/16,"motor velocity mismatch");
    }
    for(int j=0;j<9;++j) {
      require(state.press_sensor_state()[j].lost()==uint32_t(100+j),"tactile lost field mismatch");
      require(state.press_sensor_state()[j].reserve()==uint32_t(200+j),"tactile reserve field mismatch");
      for(int i=0;i<12;++i)
        require(state.press_sensor_state()[j].pressure()[i]==float(j*12+i),"tactile sensor alignment mismatch");
    }
    require(state.imu_state().quaternion()[0]==1 && state.imu_state().temperature()==28,"IMU alignment mismatch");
    require(state.power_v()==24 && state.power_a()==.5f && state.system_v()==5 && state.device_v()==3.3f,"power alignment mismatch");
    require(state.error()[0]==0 && state.error()[1]==0,"hand errors decoded at the wrong offset");
    require(state.reserve()[0]==0x12345678 && state.reserve()[1]==0x87654321,"packet tail alignment mismatch");
    std::vector<uint8_t> encoded(bytes.size());
    cdr::basic_cdr_stream writer(cdr::endianness::little_endian);
    writer.set_buffer(encoded.data(),encoded.size());
    require(cdr::write(writer,state,false) && writer.position()==bytes.size(),"HandState encode failed");
    require(encoded==bytes,"HandState re-encoded bytes differ from robot SDK format");

    // Complete XTypes identifier independently checked with the locally installed
    // Unitree Python SDK. Updating only the nested struct leaves discovery broken.
    const uint8_t identifier[]={0xf2,0xb9,0xf2,0x80,0x69,0x66,0x06,0x0f,0xc0,0xf9,0x13,0xa4,0x7b,0x8a,0xb9};
    const auto* info=Traits::type_info_blob();
    auto end=info+Traits::type_info_blob_sz();
    require(std::search(info,end,std::begin(identifier),std::end(identifier))!=end,"HandState discovery type identifier differs from robot SDK");
    std::cout<<"Dex3 robot XCDR1 layout and discovery type identifier passed\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
