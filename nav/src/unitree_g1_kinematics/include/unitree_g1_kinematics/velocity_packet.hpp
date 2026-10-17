#pragma once

#include <cstddef>
#include <cstdint>

namespace unitree_g1_kinematics
{
constexpr std::uint32_t kVelocityPacketMagic = 0x47315643U;  // "G1VC"
constexpr std::uint16_t kVelocityPacketVersion = 3;
constexpr std::uint8_t kVelocityCommand = 0;
constexpr std::uint8_t kMotionGateClosed = 1;
constexpr std::uint8_t kVelocityTimeout = 2;

struct VelocityPacket
{
  std::uint32_t magic = kVelocityPacketMagic;
  std::uint16_t version = kVelocityPacketVersion;
  std::uint8_t enabled = 0;
  std::uint8_t reason = kMotionGateClosed;
  std::uint64_t sequence = 0;
  float vx = 0.0F;
  float vy = 0.0F;
  float vyaw = 0.0F;
  // Low bit: cart frame. Other bits identify the executor's mode generation.
  std::uint32_t navigation_epoch = 0;
  // Host CLOCK_MONOTONIC time of the original /cmd_vel reception, not the
  // periodic UDP transmission. Repeating a packet must not renew its lease.
  std::uint64_t command_time_ns = 0;
};

// nav/locomotion/g1_locomotion_executor.py decodes this as '<IHBBQfffIQ'.
static_assert(sizeof(VelocityPacket) == 40, "VelocityPacket layout changed");
static_assert(offsetof(VelocityPacket, command_time_ns) == 32, "Timestamp offset changed");
}  // namespace unitree_g1_kinematics
