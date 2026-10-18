// Balance/Homie profile selection and PD handover share one SDK2 publisher.
#include "control_core.hpp"
#include "hand_core.hpp"
#include "hand_command.hpp"
#include "command_trace.hpp"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include "hand_feedback.hpp"
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

using namespace decoupled;
using LowState=unitree_hg::msg::dds_::LowState_;
using LowCmd=unitree_hg::msg::dds_::LowCmd_;
using HandCmd=unitree_hg::msg::dds_::HandCmd_;
using HandState=unitree_hg::msg::dds_::HandState_;
using Remote=unitree_go::msg::dds_::WirelessController_;
volatile std::sig_atomic_t interrupted=0;
void on_signal(int) { interrupted=1; }
uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
void print_health(const ControlCore& core,const Feedback& feedback,uint64_t now) {
  std::string detail;
  bool healthy=core.healthy(feedback,now,&detail);
  auto age_ms=[now](uint64_t received_ns) {
    return received_ns?(double(now)-double(received_ns))*1e-6:-1.0;
  };
  std::cout<<"HEALTH "<<(healthy?"true":"false")
           <<(healthy?"":" reason="+detail)
           <<" lowstate_age_ms="<<age_ms(feedback.packet.received_ns)
           <<" remote_age_ms="<<age_ms(feedback.remote_received_ns)
           <<" remote_heartbeat="<<(config::require_remote_heartbeat?"required":"ignored")
           <<" state_sequence="<<feedback.packet.sequence<<std::endl;
}
void print_joint_bounds(const char* kind,uint32_t mask,uint32_t previous,const float* q,double margin) {
  for(int i=0;i<29;++i) if((mask|previous)&(1U<<i))
    std::cout<<"BODY "<<kind<<((mask&(1U<<i))?" warning":" recovered")
             <<" index="<<i<<" q="<<q[i]<<" lower="<<config::joint_lower_sdk29[i]
             <<" upper="<<config::joint_upper_sdk29[i]<<" margin="<<margin<<std::endl;
}
struct Options {
  std::string interface;
  bool control=false,release=false,simulate=false,scripted=false,single_waist=false;
  bool prepare_full_body=false,dex3=false;
  std::array<float,14> hand_grasp{};
  std::optional<std::array<float,14>> hand_zero;
  HandPd hand_pd;
  double duration=0;
};
Options options(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg=="--control") o.control=true;
    else if(arg=="--release-mode") o.release=true;
    else if(arg=="--simulate") o.simulate=true;
    else if(arg=="--single-waist") o.single_waist=true;
    else if(arg=="--prepare-full-body") o.prepare_full_body=true;
    else if(arg=="--dex3-pd" && i+1<argc) o.hand_pd=HandPd::parse(argv[++i]);
    else if((arg=="--dex3-grasp" || arg=="--dex3-zero-compensated") && i+1<argc) {
      if(arg=="--dex3-grasp") o.dex3=true;
      else o.hand_zero=std::array<float,14>{};
      std::istringstream values(argv[++i]); std::string value;
      auto& angles=arg=="--dex3-grasp"?o.hand_grasp:*o.hand_zero;
      for(auto& q:angles) {
        if(!std::getline(values,value,',')) throw std::runtime_error("Expected 14 Dex3 angles");
        size_t used=0; q=std::stof(value,&used);
        if(used!=value.size()) throw std::runtime_error("Invalid Dex3 angle");
      }
      if(std::getline(values,value,',')) throw std::runtime_error("Too many Dex3 angles");
    }
    else if(arg=="--scripted") o.scripted=true;
    else if(arg=="--interface" && i+1<argc) o.interface=argv[++i];
    else if(arg=="--duration" && i+1<argc) o.duration=std::stod(argv[++i]);
    else if(arg=="--help") {
      std::cout<<"g1_control --interface eth0 [--control --release-mode]\n"
                 "Default: observe only, no LowCmd publisher and no mode switch.\n"
                 "Offline: --simulate [--control --scripted] [--duration N]\n"
                 "Single-waist G1: --single-waist disables SDK motor slots 13/14.\n"
                 "Viser: --prepare-full-body uses PREPARE packet targets for all active joints.\n"
                 "--dex3-grasp <14 comma-separated radians> enables coordinated hand control.\n"
                 "--dex3-pd <kp,kd> sets active hand gains (Viser passes its JSON config).\n"
                 "--dex3-zero-compensated <14 comma-separated radians> sets offsets for separate compensated open/grasp targets.\n";
      std::exit(0);
    } else throw std::runtime_error("Unknown/incomplete argument: "+arg);
  }
  if(!o.simulate && o.interface.empty()) throw std::runtime_error("--interface is required");
  if(o.release && (!o.control || o.simulate)) throw std::runtime_error("--release-mode requires real --control");
  if(o.scripted && !o.simulate) throw std::runtime_error("--scripted is allowed ONLY in --simulate");
  if(!std::isfinite(o.duration) || o.duration<0) throw std::runtime_error("Invalid duration");
  if(o.hand_zero && !o.dex3) throw std::runtime_error("--dex3-zero-compensated requires --dex3-grasp");
  HandCore validate(o.dex3,o.hand_grasp,o.hand_zero);
  return o;
}

class Udp {
 public:
  int fd=-1;
  sockaddr_in destination{};
  Udp() {
    fd=socket(AF_INET,SOCK_DGRAM,0);
    if(fd<0) throw std::runtime_error("socket failed");
    sockaddr_in local{}; local.sin_family=AF_INET; local.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    local.sin_port=htons(static_cast<uint16_t>(config::bridge_port));
    // 不启用地址复用：同一台主机只能运行一个本部署桥接。
    if(bind(fd,reinterpret_cast<sockaddr*>(&local),sizeof(local))<0) {
      close(fd); throw std::runtime_error("Bridge port is occupied");
    }
    fcntl(fd,F_SETFL,O_NONBLOCK);
    destination=local; destination.sin_port=htons(static_cast<uint16_t>(config::state_port));
  }
  ~Udp() { if(fd>=0) close(fd); }
  bool receive(CommandPacket& command) {
    sockaddr_in sender{}; socklen_t length=sizeof(sender);
    // 多读一个字节，避免把超长数据报误认为完整合法命令。
    char buffer[sizeof(command)+1];
    auto n=recvfrom(fd,buffer,sizeof(buffer),0,reinterpret_cast<sockaddr*>(&sender),&length);
    if(n<0) return false;
    if(n==sizeof(command) && sender.sin_addr.s_addr==htonl(INADDR_LOOPBACK) &&
       sender.sin_port==htons(static_cast<uint16_t>(config::state_port))) {
      std::memcpy(&command,buffer,sizeof(command)); return true;
    }
    command.magic=0; return true;
  }
  void send(const StatePacket& packet) {
    sendto(fd,&packet,sizeof(packet),0,reinterpret_cast<sockaddr*>(&destination),sizeof(destination));
  }
};

class Robot {
 public:
  explicit Robot(const Options& opt): opt_(opt) {
    if(!config::require_remote_heartbeat)
      std::cout<<"REMOTE heartbeat ignored: assuming connected; latest received buttons and sticks are held"<<std::endl;
    if(opt.simulate) return;
    unitree::robot::ChannelFactory::Instance()->Init(0,opt.interface);
    subscriber_=std::make_unique<unitree::robot::ChannelSubscriber<LowState>>("rt/lowstate");
    subscriber_->InitChannel([this](const void* message){ state(message); },1);
    remote_=std::make_unique<unitree::robot::ChannelSubscriber<Remote>>("rt/wirelesscontroller");
    remote_->InitChannel([this](const void* message){ remote(message); },1);
    if(opt.dex3) for(int side=0;side<2;++side) for(int source=0;source<2;++source) {
      auto topic=HandFeedbackSources::topic(side,source);
      auto& subscriber=hand_subscribers_[side*2+source];
      try {
        subscriber=std::make_unique<unitree::robot::ChannelSubscriber<HandState>>(topic);
        subscriber->InitChannel([this,side,source](const void* message){ hand_state(message,side,source); },1);
        std::cout<<"DEX3 SUBSCRIBE "<<topic<<std::endl;
      } catch(const std::exception& e) {
        subscriber.reset(); hand_transport_log(side,"subscribe"+std::to_string(source),e.what());
      } catch(...) { subscriber.reset(); hand_transport_log(side,"subscribe"+std::to_string(source),"unknown SDK exception"); }
    }
    if(opt.control) {
      // 按配置检查反馈后再释放原控制器；无 LowState 时保留原控制权。
      ControlCore probe(0);
      auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(config::startup_timeout_s);
      auto next_health_log=std::chrono::steady_clock::now();
      std::cout<<"CONTROL: waiting for healthy feedback before mode release; remote_heartbeat="
               <<(config::require_remote_heartbeat?"required":"ignored")<<std::endl;
      for(;;) {
        auto current=feedback(0,Output{});
        std::string health_detail;
        if(probe.healthy(current,monotonic_ns(),&health_detail)) break;
        if(std::chrono::steady_clock::now()>=next_health_log) {
          print_health(probe,current,monotonic_ns());
          next_health_log=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        }
        if(interrupted || std::chrono::steady_clock::now()>=deadline)
          throw std::runtime_error("Control preflight failed; factory mode was not released");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      // 只读入口绝不调用模式切换或创建 LowCmd publisher。
      unitree::robot::b2::MotionSwitcherClient switcher;
      switcher.SetTimeout(2.0f); switcher.Init();
      std::string form,name;
      for(int attempt=0;attempt<3;++attempt) {
        if(switcher.CheckMode(form,name)!=0) throw std::runtime_error("CheckMode failed; no motor commands sent");
        if(name.empty()) break;
        if(!opt.release) throw std::runtime_error("Factory controller active; use --release-mode explicitly");
        if(switcher.ReleaseMode()!=0) throw std::runtime_error("ReleaseMode failed; no motor commands sent");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
      if(switcher.CheckMode(form,name)!=0 || !name.empty())
        throw std::runtime_error("Could not verify exclusive low-level control");
      publisher_=std::make_unique<unitree::robot::ChannelPublisher<LowCmd>>("rt/lowcmd");
      publisher_->InitChannel();
      if(opt.dex3) for(int side=0;side<2;++side) {
        std::string topic=side==0?"rt/dex3/left/cmd":"rt/dex3/right/cmd";
        try {
          hand_publishers_[side]=std::make_unique<unitree::robot::ChannelPublisher<HandCmd>>(topic);
          hand_publishers_[side]->InitChannel();
        } catch(const std::exception& e) {
          hand_publishers_[side].reset(); hand_transport_log(side,"publish_init",e.what());
        } catch(...) { hand_publishers_[side].reset(); hand_transport_log(side,"publish_init","unknown SDK exception"); }
      }
    }
  }
  Feedback feedback(double elapsed,const Output& previous) {
    if(opt_.simulate) {
      uint64_t now=monotonic_ns();
      Feedback f; f.valid=true; f.remote_valid=true;
      f.remote_received_ns=now;
      f.packet.sequence=++sim_seq_; f.packet.received_ns=now; f.packet.quaternion[0]=1;
      f.packet.mode_machine=1;
      for(int i=0;i<29;++i) {
        if(!sim_initialized_) sim_q_[i]=config::default_q_sdk29[i];
        if(previous.publish && previous.kp[i]>0) sim_q_[i]+=0.15f*(previous.q[i]-sim_q_[i]);
        f.packet.q[i]=sim_q_[i];
      }
      sim_initialized_=true;
      if(opt_.scripted) {
        if((elapsed>1.0 && elapsed<1.25) || (elapsed>5.0 && elapsed<5.25)) f.packet.buttons|=kR2;
        // 只在合成反馈模式注入速度，联调覆盖 Balance→Walk→Balance。
        if(elapsed>6.0 && elapsed<6.8) { f.packet.buttons|=1U<<1; f.packet.sticks[1]=1.0f; }
        if(elapsed>7.5) f.packet.buttons|=kL2;
      }
      return f;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto f=latest_;
    // 回调可能在等待锁时更新接收时间；必须取得快照后再读当前时间。
    uint64_t now=monotonic_ns();
    f.remote_received_ns=remote_ns_;
    f.remote_valid=remote_ns_ && now>=remote_ns_ && (now-remote_ns_)*1e-9<config::state_timeout_s;
    f.packet.buttons|=remote_buttons_;
    std::copy(sticks_.begin(),sticks_.end(),std::begin(f.packet.sticks));
    return f;
  }
  void write(const Output& output,uint32_t mode_machine) {
    if(!output.publish || (!opt_.simulate && !publisher_)) return;
    LowCmd command;
    command.mode_pr()=0; command.mode_machine()=static_cast<uint8_t>(mode_machine);
    for(int i=0;i<29;++i) {
      auto& motor=command.motor_cmd()[i];
      if(opt_.single_waist && (i==13 || i==14)) {
        motor.mode()=0; motor.q()=0; motor.dq()=0;
        motor.kp()=0; motor.kd()=0; motor.tau()=0;
        continue;
      }
      motor.mode()=1; motor.q()=output.q[i]; motor.dq()=0;
      motor.kp()=output.kp[i]; motor.kd()=output.kd[i]; motor.tau()=0;
    }
    command.crc()=crc32(&command,sizeof(command)-sizeof(uint32_t));
    if(!opt_.simulate && !publisher_->Write(command)) throw std::runtime_error("DDS LowCmd publish failed");
    trace_.published(0,command.motor_cmd(),monotonic_ns());
  }
  HandFeedback hand_feedback(const HandCore& hands) {
    if(opt_.simulate) {
      HandFeedback f; f.valid.fill(true); f.received_ns.fill(monotonic_ns());
      if(hands.ever_enabled) { f.q=hands.q; f.dq=hands.dq; }
      return f;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    return hand_feedback_sources_.snapshot(monotonic_ns());
  }
  void write_hands(HandCore& hands) {
    const bool was_enabled=hands.active || hands.recovering;
    auto errors=publish_hand_commands<HandCmd>(hands,opt_.hand_pd,[&](int side,const HandCmd& command) {
      if(!opt_.simulate && (!hand_publishers_[side] || !hand_publishers_[side]->Write(command))) return false;
      trace_.published(side+1,command.motor_cmd(),monotonic_ns());
      if(last_hand_active_[side]!=int(hands.active)) {
        std::cout<<"DEX3 PD side="<<(side==0?"left":"right")
                 <<" active="<<(hands.active?"true":"false")
                 <<" kp="<<command.motor_cmd()[0].kp()<<" kd="<<command.motor_cmd()[0].kd()
                 <<" write="<<(opt_.simulate?"synthetic":"dds_success")<<std::endl;
        last_hand_active_[side]=int(hands.active);
      }
      return true;
    });
    for(int side=0;side<2;++side) if(errors[side]!=last_hand_write_errors_[side]) {
      hand_transport_log(side,"publish",errors[side]);
      last_hand_write_errors_[side]=errors[side];
    }
    if(was_enabled && !hands.active && !hands.recovering)
      std::cout<<"DEX3 CONTROL warning reason=hand command transport failed; body control continues"<<std::endl;
  }
  void telemetry(StatePacket& packet) const { trace_.telemetry(packet,monotonic_ns()); }
 private:
  CommandTrace trace_;
  std::array<int,2> last_hand_active_{{-1,-1}};
  std::array<std::string,2> last_hand_write_errors_;
  std::array<std::array<uint64_t,2>,2> last_bad_hand_sample_log_{};
  static void hand_transport_log(int side,const std::string& operation,const std::string& error) {
    std::cout<<"DEX3 TRANSPORT side="<<(side==0?"left":"right")<<" operation="<<operation
             <<(error.empty()?" recovered":" warning reason="+error)<<std::endl;
  }
  void hand_state(const void* message,int side,int source) {
    const auto& msg=*static_cast<const HandState*>(message);
    std::lock_guard<std::mutex> guard(mutex_);
    auto& feedback=hand_feedback_sources_.samples[source];
    const auto now=monotonic_ns();
    if(!receive_hand_feedback(msg,side,now,config::max_temperature_c,feedback)) {
      auto& logged=last_bad_hand_sample_log_[source][side];
      if(!logged || now-logged>=1000000000ULL) {
        std::cout<<"DEX3 SAMPLE discarded side="<<(side==0?"left":"right")
                 <<" topic="<<HandFeedbackSources::topic(side,source)
                 <<" motor_count="<<msg.motor_state().size()
                 <<"; last complete feedback age retained"<<std::endl;
        logged=now;
      }
    }
  }
  void state(const void* message) {
    auto& msg=*static_cast<const LowState*>(message);
    if(msg.crc()!=crc32(&msg,sizeof(msg)-sizeof(uint32_t))) return;
    std::lock_guard<std::mutex> guard(mutex_);
    // Duplicate/backward ticks cannot refresh feedback age. Unsigned subtraction
    // accepts normal uint32 wrap; a sensor tick reset expires the shared watchdog.
    const uint32_t tick_delta=msg.tick()-last_tick_;
    if(latest_.valid && (tick_delta==0 || tick_delta>0x7fffffffu)) return;
    last_tick_=msg.tick();
    Feedback next; next.valid=true; next.packet.sequence=latest_.packet.sequence+1;
    next.packet.received_ns=monotonic_ns(); next.packet.mode_machine=msg.mode_machine();
    for(int i=0;i<29;++i) {
      next.packet.q[i]=msg.motor_state()[i].q(); next.packet.dq[i]=msg.motor_state()[i].dq();
      next.motor_temperature[i]=msg.motor_state()[i].temperature();
      if(!opt_.single_waist || (i!=13 && i!=14)) {
        next.motor_error=next.motor_error || msg.motor_state()[i].motorstate()!=0;
      }
    }
    std::copy(msg.imu_state().quaternion().begin(),msg.imu_state().quaternion().end(),std::begin(next.packet.quaternion));
    std::copy(msg.imu_state().gyroscope().begin(),msg.imu_state().gyroscope().end(),std::begin(next.packet.omega));
    std::copy(msg.imu_state().rpy().begin(),msg.imu_state().rpy().end(),std::begin(next.packet.rpy));
    // L2 同时从 LowState 的原始遥控字段获取，Python 或另一个遥控订阅失效也可触发。
    next.packet.buttons=(uint32_t(msg.wireless_remote()[2]) | (uint32_t(msg.wireless_remote()[3])<<8))&kL2;
    latest_=next;
  }
  void remote(const void* message) {
    auto& msg=*static_cast<const Remote*>(message);
    std::lock_guard<std::mutex> guard(mutex_);
    sticks_={msg.lx(),msg.ly(),msg.rx(),msg.ry()}; remote_buttons_=msg.keys(); remote_ns_=monotonic_ns();
  }
  Options opt_;
  std::mutex mutex_;
  Feedback latest_;
  HandFeedbackSources hand_feedback_sources_;
  uint32_t last_tick_=0,remote_buttons_=0;
  uint64_t remote_ns_=0,sim_seq_=0;
  std::array<float,4> sticks_{};
  std::array<float,29> sim_q_{};
  bool sim_initialized_=false;
  std::unique_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<Remote>> remote_;
  std::unique_ptr<unitree::robot::ChannelPublisher<LowCmd>> publisher_;
  std::array<std::unique_ptr<unitree::robot::ChannelSubscriber<HandState>>,4> hand_subscribers_;
  std::array<std::unique_ptr<unitree::robot::ChannelPublisher<HandCmd>>,2> hand_publishers_;
};

int main(int argc,char** argv) {
  try {
    auto opt=options(argc,argv);
    std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);
    Udp udp; Robot robot(opt);
    uint64_t session=(uint64_t(std::random_device{}())<<32)^monotonic_ns();
    ControlCore core(session,config::require_remote_heartbeat,opt.prepare_full_body,opt.single_waist);
    HandCore hands(opt.dex3,opt.hand_grasp,opt.hand_zero);
    Output output; Feedback feedback;
    auto start=std::chrono::steady_clock::now(),next=start;
    uint64_t stop_started=0;
    uint64_t last_health_log=0;
    uint64_t last_position_log=0,last_target_log=0,last_effort_log=0;
    uint32_t previous_position_warning=0,previous_target_warning=0,previous_effort_warning=0;
    uint64_t last_hand_health_log=0;
    uint32_t previous_hand_health=0;
    uint32_t previous_hand_position_warning=0;
    std::array<int,2> previous_hand_source{{-1,-1}};
    uint64_t last_calibration_log=0;
    bool previous_preparation_warning=false;
    bool previous_health=true;
    Stage previous=Stage::Waiting;
    std::cout<<"BRIDGE mode="<<(opt.simulate?"SIMULATION":opt.control?"CONTROL_ARMED":"OBSERVE_ONLY")
             <<" config="<<config::token<<std::endl;
    for(;;) {
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      feedback=robot.feedback(elapsed,output);
      auto hand_feedback=robot.hand_feedback(hands);
      // 使用晚于快照的时间，避免把刚到的反馈误判为来自未来。
      uint64_t now=monotonic_ns();
      const uint32_t position_warning=core.position_warning_mask(feedback);
      if(position_warning!=previous_position_warning ||
         (position_warning && (now-last_position_log)*1e-9>=1.)) {
        print_joint_bounds("POSITION",position_warning,previous_position_warning,feedback.packet.q,.02);
        last_position_log=now; previous_position_warning=position_warning;
      }
      uint32_t hand_health=hand_feedback.healthy_flags(now);
      uint32_t hand_position_warning=hand_feedback.position_warning_mask();
      if(opt.dex3 && (!last_hand_health_log || hand_health!=previous_hand_health ||
          hand_position_warning!=previous_hand_position_warning ||
          hand_feedback.source!=previous_hand_source ||
          ((hand_health!=3 || hand_position_warning) && (now-last_hand_health_log)*1e-9>=1.))) {
        for(int side=0;side<2;++side) {
          auto problem=hand_feedback.problem(side,now);
          std::cout<<"DEX3 HEALTH side="<<(side==0?"left":"right")
                   <<" healthy="<<((hand_health&(1U<<side))?"true":"false")
                   <<" topic="<<(hand_feedback.source[side]<0?"none":
                                  HandFeedbackSources::topic(side,hand_feedback.source[side]))
                   <<" age_ms="<<(hand_feedback.received_ns[side]?
                                    (double(now)-double(hand_feedback.received_ns[side]))*1e-6:-1.)
                   <<(problem.empty()?"":" reason="+problem)<<std::endl;
        }
        last_hand_health_log=now;
        previous_hand_health=hand_health;
        previous_hand_position_warning=hand_position_warning;
        previous_hand_source=hand_feedback.source;
      }
      CommandPacket command;
      // 每个周期有界处理数据报，不允许本机流量饿死 200 Hz 看门狗。
      if(hands.check_feedback(hand_feedback,now))
        std::cout<<"DEX3 CONTROL warning reason=feedback unhealthy or stale; body control continues"<<std::endl;
      for(int i=0;i<64 && udp.receive(command);++i) {
        if(!opt.control) continue;
        Stage before=core.stage;
        if(!core.command(command,feedback,now)) continue;
        const uint32_t target_warning=core.target_warning_mask();
        if(target_warning!=previous_target_warning ||
           (target_warning && (now-last_target_log)*1e-9>=1.)) {
          print_joint_bounds("TARGET",target_warning,previous_target_warning,command.q,0);
          last_target_log=now; previous_target_warning=target_warning;
        }
        const bool hand_was_active=hands.active;
        if(before==Stage::Waiting && core.stage==Stage::Ramping && opt.dex3) {
          if(hands.recovering) {
            std::cout<<"DEX3 RECOVERY warning joint="<<hands.recovery_joint
                     <<" reason=body preparation requested"<<std::endl;
            hands.cancel_recovery("body preparation requested");
          }
          // Body/arm preparation must not enable or zero the fingers. A later
          // explicit hand pose request starts from fresh measured hand angles.
          hands.suspend();
        }
        if(!(feedback.packet.buttons & (kB|kL2))) {
          hands.command(command,core.stage,now,hand_feedback);
          if(command.request==uint32_t(Request::Wait) &&
             hands.request_recovery(command.hand_recover,core.stage,hand_feedback,now))
            std::cout<<"DEX3 RECOVERY started joint="<<hands.recovery_joint
                     <<" initial_q="<<hand_feedback.q[hands.recovery_joint]
                     <<" target_q="<<hands.q[hands.recovery_joint]
                     <<" kp_min="<<hand_recovery_kp_min<<" kp_max="<<hand_recovery_kp_max
                     <<" timeout_s="<<hand_recovery_timeout_s<<std::endl;
        }
        if(!hand_was_active && hands.active) std::cout<<"DEX3 CONTROL resumed"<<std::endl;
      }
      if(interrupted || (opt.duration>0 && elapsed>=opt.duration)) core.stop("process shutdown",false);
      if(hands.recovering) {
        std::string detail;
        if(!core.healthy(feedback,now,&detail)) core.stop("Dex3 recovery: body feedback unhealthy: "+detail);
        if(feedback.packet.buttons & (kB|kL2)) core.stop("Dex3 recovery: remote stop",false);
        const int recovered_joint=hands.recovery_joint;
        auto error=hands.step_recovery(hand_feedback,now);
        if(!error.empty()) std::cout<<"DEX3 RECOVERY warning joint="<<recovered_joint
                                  <<" reason="<<error<<std::endl;
        else if(!hands.recovering) std::cout<<"DEX3 RECOVERY completed joint="<<recovered_joint<<std::endl;
      }
      Stage stage_before_step=core.stage;
      output=core.step(feedback,now);
      const uint32_t effort_warning=core.effort_warning_mask();
      if(effort_warning!=previous_effort_warning ||
         (effort_warning && (now-last_effort_log)*1e-9>=1.)) {
        for(int i=0;i<29;++i) if((effort_warning|previous_effort_warning)&(1U<<i))
          std::cout<<"BODY EFFORT"<<((effort_warning&(1U<<i))?" warning":" recovered")
                   <<" index="<<i<<" kp="<<output.kp[i]<<" kd="<<output.kd[i]<<std::endl;
        last_effort_log=now; previous_effort_warning=effort_warning;
      }
      hands.step(now,core.stage==Stage::Damped || core.stage==Stage::Fault);
      // The outer console's physical G1 has fixed waist roll/pitch (SDK slots 13/14).
      // Keep SDK numbering and model inputs, but never enable these absent motors.
      if(opt.single_waist) for(int i: {13,14}) {
        output.q[i]=0; output.kp[i]=0; output.kd[i]=0;
      }
      try { if(opt.control) robot.write(output,feedback.packet.mode_machine); }
      catch(const std::exception& e) {
        core.stop(e.what()); output=core.step(feedback,now); hands.step(now,true);
      }
      if(opt.control) robot.write_hands(hands);
      if(opt.single_waist) for(int i: {13,14}) {
        output.q[i]=0; output.kp[i]=0; output.kd[i]=0;
      }
      StatePacket packet=feedback.packet;
      hands.telemetry(packet,hand_feedback,now);
      std::copy(hand_feedback.received_ns.begin(),hand_feedback.received_ns.end(),packet.hand_received_ns);
      robot.telemetry(packet);
      packet.session=session; packet.config_token=config::token; packet.stage=static_cast<uint32_t>(core.stage);
      bool healthy=core.healthy(feedback,now);
      const bool preparation_warning=core.preparation_overdue(now);
      packet.flags=(healthy?1U:0U) | (opt.control?2U:0U) | (opt.simulate?4U:0U) |
                   (preparation_warning?kPreparationOverdue:0U);
      for(int i=0;i<29;++i) packet.applied_q[i]=output.publish?output.q[i]:feedback.packet.q[i];
      packet.profile=output.profile; packet.gain_blend=output.gain_blend;
      packet.upper_mode=output.upper_mode; packet.upper_gain_blend=output.upper_gain_blend;
      for(int i=0;i<29;++i) {
        packet.applied_kp[i]=output.kp[i]; packet.applied_kd[i]=output.kd[i];
      }
      udp.send(packet);
      // 输出机身健康检查与准备过程诊断。
      if((!healthy && (!last_health_log || (now-last_health_log)*1e-9>=1.0)) ||
         (healthy && !previous_health)) {
        print_health(core,feedback,now);
        last_health_log=now;
      }
      previous_health=healthy;
      if(stage_before_step==Stage::Ramping && core.stage!=Stage::Fault &&
         (!last_calibration_log || preparation_warning!=previous_preparation_warning ||
          (now-last_calibration_log)*1e-9>=1.0 || core.stage==Stage::Holding)) {
        std::cout<<"CALIBRATION "<<(preparation_warning?"warning holding_position_control ":"")
                 <<core.calibration_diagnostics(feedback,now)<<std::endl;
        last_calibration_log=now;
      }
      previous_preparation_warning=preparation_warning;
      if(core.stage!=previous) {
        std::cout<<"STAGE "<<static_cast<int>(core.stage)<<" "<<core.reason<<std::endl; previous=core.stage;
      }
      if(core.stage==Stage::Fault || core.stage==Stage::Damped) {
        if(!stop_started) stop_started=now;
        if((!core.ever_enabled && !hands.ever_enabled) || (now-stop_started)*1e-9>=config::shutdown_damping_s) break;
      }
      next+=std::chrono::microseconds(5000);
      if(next<std::chrono::steady_clock::now()) next=std::chrono::steady_clock::now();
      std::this_thread::sleep_until(next);
    }
    return core.stage==Stage::Fault?2:0;
  } catch(const std::exception& e) {
    std::cerr<<"BRIDGE ERROR: "<<e.what()<<std::endl; return 1;
  }
}
