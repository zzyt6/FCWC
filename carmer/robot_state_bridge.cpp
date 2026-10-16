// Read-only DDS subscriber. No command publisher, mode client or actuation API.
// JSON lines to the parent; CLOCK_MONOTONIC is shared with Python on this host.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <time.h>

namespace {
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }
uint64_t monotonic_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
uint32_t crc32(const void* data, size_t bytes) {
  uint32_t crc = 0xffffffff;
  auto p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; i += 4) {
    uint32_t word;
    std::memcpy(&word, p + i, 4);
    for (uint32_t bit = 0x80000000; bit; bit >>= 1) {
      bool high = crc & 0x80000000;
      crc <<= 1;
      if (high) crc ^= 0x04c11db7;
      if (word & bit) crc ^= 0x04c11db7;
    }
  }
  return crc;
}
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: robot_state_bridge NETWORK_INTERFACE\n";
    return 2;
  }
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    using State = unitree_hg::msg::dds_::LowState_;
    std::mutex mutex;
    std::string latest;
    uint32_t previous_tick = 0;
    bool received = false;
    unitree::robot::ChannelSubscriber<State> subscriber("rt/lowstate");
    subscriber.InitChannel([&](const void* data) {
      const auto& state = *static_cast<const State*>(data);
      const auto stamp = monotonic_ns();
      if (state.crc() != crc32(&state, sizeof(state) - sizeof(uint32_t))) return;
      std::lock_guard<std::mutex> guard(mutex);
      const uint32_t delta = state.tick() - previous_tick;
      if (received && (delta == 0 || delta > 0x7fffffffU)) return;
      previous_tick = state.tick();
      received = true;
      std::ostringstream line;
      line << std::setprecision(10)
           << "{\"received_ns\":" << stamp << ",\"tick\":" << state.tick()
           << ",\"mode_pr\":" << unsigned(state.mode_pr()) << ",\"q\":[";
      for (size_t i = 0; i < 15; ++i) {
        if (i) line << ',';
        // Locked waist roll/pitch channels are not kinematic inputs.
        const float value = i < 13 ? state.motor_state()[i].q() : 0.f;
        if (std::isfinite(value)) line << value;
        else line << "null";
      }
      line << "],\"quaternion\":[";
      for (size_t i = 0; i < 4; ++i) {
        if (i) line << ',';
        const float value = state.imu_state().quaternion()[i];
        if (std::isfinite(value)) line << value;
        else line << "null";
      }
      line << "]}";
      latest = line.str();
    }, 1);
    // The DDS callback never waits for stdout. Only the newest packet is sent.
    while (running && std::cout.good()) {
      std::string packet;
      {
        std::lock_guard<std::mutex> guard(mutex);
        packet.swap(latest);
      }
      if (!packet.empty()) std::cout << packet << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
    subscriber.CloseChannel();
  } catch (const std::exception& error) {
    std::cerr << "Robot feedback: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
