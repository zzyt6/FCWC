#pragma once

#include <array>
#include <cstdint>

namespace unitree_g1_kinematics
{
constexpr std::uint32_t kLowStatePacketMagic = 0x47314c53U;  // "G1LS"
constexpr std::uint16_t kLowStatePacketVersion = 2;

struct LowStatePacket
{
  std::uint32_t magic = kLowStatePacketMagic;
  std::uint16_t version = kLowStatePacketVersion;
  std::uint8_t mode_machine = 0;
  std::uint8_t reserved = 0;
  std::uint64_t sequence = 0;
  double sample_monotonic = 0.;
  std::array<double, 29> q{};
};
}  // namespace unitree_g1_kinematics
