#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>

namespace decoupled {
// Optional read-only tail of the existing state datagram. These fields never
// participate in motor commands or health decisions.
constexpr uint32_t kMotorTelemetryMagic=0x4d545231, kMotorTelemetryVersion=1;
#pragma pack(push, 1)
struct BodyTelemetry {
  uint32_t available=0, tick=0, mode_pr=0, mode[29]{};
  float ddq[29]{}, tau_est[29]{}, vol[29]{}, acc[3]{};
};
struct HandTelemetry {
  uint32_t available=0;
  int32_t source=-1;
  uint32_t press_sensor_count=0;
  float ddq[7]{}, tau_est[7]{}, power[4]{}, pressure[108]{};
};
struct MotorTelemetry {
  uint32_t magic=kMotorTelemetryMagic, version=kMotorTelemetryVersion;
  BodyTelemetry body;
  HandTelemetry hands[2];
};
#pragma pack(pop)
static_assert(sizeof(MotorTelemetry)==1528, "Motor telemetry wire format changed");

template<class Message>
BodyTelemetry body_telemetry(const Message& msg) {
  BodyTelemetry out;
  out.available=1; out.tick=msg.tick(); out.mode_pr=msg.mode_pr();
  for(int i=0;i<29;++i) {
    const auto& m=msg.motor_state()[i];
    out.mode[i]=m.mode(); out.ddq[i]=m.ddq(); out.tau_est[i]=m.tau_est(); out.vol[i]=m.vol();
  }
  std::copy(msg.imu_state().accelerometer().begin(),msg.imu_state().accelerometer().end(),out.acc);
  return out;
}

template<class Message>
HandTelemetry hand_telemetry(const Message& msg,int source) {
  HandTelemetry out;
  if(msg.motor_state().size()!=7) return out;
  out.available=1; out.source=source;
  for(int i=0;i<7;++i) {
    out.ddq[i]=msg.motor_state()[i].ddq(); out.tau_est[i]=msg.motor_state()[i].tau_est();
  }
  out.power[0]=msg.power_v(); out.power[1]=msg.power_a();
  out.power[2]=msg.system_v(); out.power[3]=msg.device_v();
  out.press_sensor_count=msg.press_sensor_state().size();
  std::fill(std::begin(out.pressure),std::end(out.pressure),std::numeric_limits<float>::quiet_NaN());
  for(uint32_t k=0;k<std::min(out.press_sensor_count,9U);++k)
    std::copy(msg.press_sensor_state()[k].pressure().begin(),
              msg.press_sensor_state()[k].pressure().end(),out.pressure+k*12);
  return out;
}
}
