#include <arpa/inet.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include "unitree_g1_kinematics/lowstate_packet.hpp"

namespace
{
std::atomic<bool> running{true};

uint32_t crc32(const void * data, size_t bytes)
{
  uint32_t crc = 0xffffffff;
  const auto * p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; i += 4) {
    uint32_t word;
    std::memcpy(&word, p+i, 4);
    for (uint32_t bit = 0x80000000; bit; bit >>= 1) {
      const bool high = crc & 0x80000000;
      crc <<= 1;
      if (high) crc ^= 0x04c11db7;
      if (word & bit) crc ^= 0x04c11db7;
    }
  }
  return crc;
}

void stopHandler(int)
{
  running = false;
}

std::string environmentOr(const char *name, const char *fallback)
{
  const char *value = std::getenv(name);
  return value && *value ? value : fallback;
}
}  // namespace

class LowStateUdpBridge
{
public:
  LowStateUdpBridge()
  {
    const std::string network_interface =
        environmentOr("G1_NETWORK_INTERFACE", "eth0");
    const std::string lowstate_topic =
        environmentOr("G1_LOWSTATE_TOPIC", "rt/lowstate");
    const std::string udp_host =
        environmentOr("G1_LOWSTATE_UDP_HOST", "127.0.0.1");
    const int udp_port = std::stoi(
        environmentOr("G1_LOWSTATE_UDP_PORT", "39001"));

    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) throw std::runtime_error("Cannot create lowstate UDP socket");
    destination_.sin_family = AF_INET;
    destination_.sin_port = htons(static_cast<std::uint16_t>(udp_port));
    if (::inet_pton(AF_INET, udp_host.c_str(), &destination_.sin_addr) != 1)
      throw std::runtime_error("Invalid G1_LOWSTATE_UDP_HOST");

    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
    subscriber_ = std::make_shared<
        unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(
        lowstate_topic);
    subscriber_->InitChannel(
        [this](const void *message) { forward(message); }, 1);

    std::cout << "Unitree SDK2 lowstate bridge: " << lowstate_topic << " on "
              << network_interface << " -> udp://" << udp_host << ':' << udp_port
              << std::endl;
  }

  ~LowStateUdpBridge()
  {
    if (subscriber_) subscriber_->CloseChannel();
    if (socket_ >= 0) ::close(socket_);
  }

private:
  void forward(const void *message)
  {
    const auto *state =
        static_cast<const unitree_hg::msg::dds_::LowState_ *>(message);
    if (state->crc() != crc32(state, sizeof(*state)-sizeof(uint32_t))) return;
    const uint32_t delta = state->tick()-last_tick_;
    if (have_tick_ && (delta == 0 || delta > 0x7fffffffU)) return;
    last_tick_ = state->tick();
    have_tick_ = true;
    unitree_g1_kinematics::LowStatePacket packet;
    packet.sample_monotonic = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    packet.sequence = sequence_++;
    packet.mode_machine = state->mode_machine();
    for (std::size_t index = 0; index < packet.q.size(); ++index)
      packet.q[index] = state->motor_state().at(index).q();
    ::sendto(socket_, &packet, sizeof(packet), MSG_DONTWAIT,
             reinterpret_cast<const sockaddr *>(&destination_), sizeof(destination_));
  }

  int socket_ = -1;
  sockaddr_in destination_{};
  std::uint64_t sequence_ = 0;
  uint32_t last_tick_ = 0;
  bool have_tick_ = false;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> subscriber_;
};

int main()
{
  ::signal(SIGINT, stopHandler);
  ::signal(SIGTERM, stopHandler);
  try
  {
    LowStateUdpBridge bridge;
    while (running) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  return 0;
}
