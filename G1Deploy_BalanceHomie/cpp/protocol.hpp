#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "motor_telemetry.hpp"

namespace decoupled {
constexpr uint32_t kStateMagic = 0x48594253, kCommandMagic = 0x48594243, kVersion = 9;
constexpr uint32_t kR2 = 1U << 4, kL2 = 1U << 5, kB = 1U << 9;
// State flags: healthy=1, control armed=2, simulation=4, preparation overdue=8.
constexpr uint32_t kPreparationOverdue = 1U << 3;
// The policy command is late; the bridge holds the last leg targets with
// Balance gains instead of damping (see ControlCore::begin_policy_hold).
constexpr uint32_t kPolicyHold = 1U << 4;
enum class Stage : uint32_t { Waiting=0, Ramping=1, Holding=2, Policy=3, Damped=4, Fault=5 };
enum class Request : uint32_t { Wait=0, Prepare=1, Hold=2, Policy=3, Stop=4 };
enum class Profile : uint32_t { Balance=0, Homie=1 };
enum class UpperMode : uint32_t { Hold=0, Pushcart=1 };
inline bool is_pushcart_upper_joint(int i) { return i==12 || (i>=15 && i<29); }

// 只在同一台 Linux 主机的 loopback 上通信，x86_64/aarch64 均为 little-endian。
#pragma pack(push, 1)
struct StatePacket {
  uint32_t magic=kStateMagic, version=kVersion;
  uint64_t session=0, sequence=0, received_ns=0, config_token=0;
  uint32_t stage=0, flags=0, buttons=0, mode_machine=0;
  float q[29]{}, dq[29]{}, omega[3]{}, quaternion[4]{}, rpy[3]{}, sticks[4]{}, applied_q[29]{};
  uint32_t profile=0;
  float gain_blend=1, applied_kp[29]{}, applied_kd[29]{};
  uint32_t hand_flags=0; // Bits 1/2: SDK/feedback health, excluding position-limit warnings.
  float hand_progress=0, hand_q[14]{}, hand_dq[14]{}, hand_target[14]{};
  // Successful local DDS writes, never an acknowledgement from a motor.
  // Groups: body, left Dex3, right Dex3. Joints: SDK29, left7, right7.
  uint64_t bridge_ns=0, sent_ns[3]{}, hand_received_ns[2]{}, position_mask=0;
  float sent_q[43]{};
  uint32_t hand_recovery_mask=0;
  int32_t hand_recovery_joint=-1;
  uint32_t upper_mode=0;
  float upper_gain_blend=1;
};
struct CommandPacket {
  uint32_t magic=kCommandMagic, version=kVersion;
  uint64_t session=0, sequence=0, observed_sequence=0, observed_received_ns=0, config_token=0;
  uint32_t request=0, profile=0;
  float q[29]{};
  uint32_t hand_pose=0; // 0: idle/acknowledge hand pause, 1: raw zero, 2: grasp, 3: compensated zero, 4: compensated grasp
  int32_t hand_recover=-1; // -1: no recovery; 0..13: selected Dex3 joint, only before prepare
  uint32_t hand_grasp_mode=0; // 0: unchanged, 1: update preset, 2: follow preset live (already enabled only)
  float hand_grasp[14]{}; // Raw radians, before optional zero compensation.
  uint32_t upper_mode=0; // 0: held pose + locomotion PD; 1: push-cart waist/arm targets + PD.
  uint32_t upper_target_active=0; // 0: cancel upper motion at last emitted target; 1: follow q in push-cart mode.
};
#pragma pack(pop)
static_assert(sizeof(StatePacket)==1120, "State wire format changed");
static_assert(sizeof(CommandPacket)==248, "Command wire format changed");

struct Feedback {
  StatePacket packet;
  BodyTelemetry motor_telemetry;
  bool valid=false;
  bool motor_error=false;
  bool remote_valid=false;
  uint64_t remote_received_ns=0;
  // Keep both SDK readings per motor so fault logs identify every hot joint.
  // Internal feedback only; the UDP StatePacket layout is unchanged.
  std::array<std::array<int16_t,2>,29> motor_temperature{};
};
struct Output {
  bool publish=false;
  std::array<float,29> q{}, kp{}, kd{};
  uint32_t profile=0;
  float gain_blend=1;
  uint32_t upper_mode=0;
  float upper_gain_blend=1;
};
inline uint32_t crc32(const void* data, size_t bytes) {
  uint32_t crc=0xffffffff;
  auto p=static_cast<const unsigned char*>(data);
  for(size_t i=0;i<bytes;i+=4) {
    uint32_t word; std::memcpy(&word,p+i,4);
    for(uint32_t bit=0x80000000;bit;bit>>=1) {
      bool high=crc&0x80000000; crc<<=1;
      if(high) crc^=0x04c11db7;
      if(word&bit) crc^=0x04c11db7;
    }
  }
  return crc;
}
}
