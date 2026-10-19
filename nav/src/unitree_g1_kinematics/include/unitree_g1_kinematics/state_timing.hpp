#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <utility>

#include "unitree_g1_kinematics/lowstate_packet.hpp"

namespace unitree_g1_kinematics
{
// The LiDAR clock is independent of the host clock. Estimate its offset from
// the low-latency IMU stream, not from completion of a LiDAR mapping callback.
// Taking the minimum over a bounded window excludes positive scheduling delay.
class SensorClock
{
public:
  void observe(double sensor, double received)
  {
    if (!std::isfinite(sensor) || !std::isfinite(received)) {return;}
    if (!samples_.empty() && sensor == last_sensor_) return;
    if (!samples_.empty() && sensor < last_sensor_) {
      // A delayed/duplicated IMU packet cannot redefine the sensor epoch.
      // A real clock restart may establish a new epoch after the old one expires.
      if (received-samples_.back().first <= .3) return;
    }
    if (!samples_.empty() && sensor > last_sensor_ && received-sensor-offset_ > 1.0) {
      return; // A large positive transport delay is not a new sensor epoch.
    }
    if (!samples_.empty() && (sensor < last_sensor_ || received < samples_.back().first ||
      (received - sensor) - offset_ < -1.0))
    {
      samples_.clear();
    }
    last_sensor_ = sensor;
    samples_.emplace_back(received, received - sensor);
    while (samples_.size() > 1 && received - samples_.front().first > 10.) {
      samples_.pop_front();
    }
    offset_ = std::min_element(samples_.begin(), samples_.end(),
      [](const auto & a, const auto & b) {return a.second < b.second;})->second;
  }

  std::optional<double> hostTime(double sensor, double now) const
  {
    if (samples_.empty() || now - samples_.back().first > .3) {return std::nullopt;}
    const double stamp = sensor + offset_;
    if (!std::isfinite(stamp) || now - stamp > .5 || stamp - now > .03) {
      return std::nullopt;
    }
    return stamp;
  }

private:
  std::deque<std::pair<double, double>> samples_;
  double last_sensor_ = 0., offset_ = 0.;
};

class JointHistory
{
public:
  bool add(const LowStatePacket & packet)
  {
    if (!std::isfinite(packet.sample_monotonic)) {return false;}
    for (const auto q : packet.q) {if (!std::isfinite(q)) {return false;}}
    if (!samples_.empty()) {
      if (packet.sample_monotonic <= samples_.back().sample_monotonic) {return false;}
      if (packet.mode_machine != samples_.back().mode_machine) {samples_.clear();}
    }
    samples_.push_back(packet);
    while (samples_.size() > 1 &&
      packet.sample_monotonic - samples_.front().sample_monotonic > 1.)
    {
      samples_.pop_front();
    }
    return true;
  }

  std::optional<LowStatePacket> at(double stamp) const
  {
    if (samples_.empty()) {return std::nullopt;}
    if (stamp < samples_.front().sample_monotonic) {return std::nullopt;}
    if (stamp >= samples_.back().sample_monotonic) {
      if (stamp - samples_.back().sample_monotonic <= .01) {return samples_.back();}
      return std::nullopt;
    }
    for (size_t i = 1; i < samples_.size(); ++i) {
      const auto & a = samples_[i - 1];
      const auto & b = samples_[i];
      if (stamp > b.sample_monotonic) {continue;}
      const double dt = b.sample_monotonic - a.sample_monotonic;
      if (dt > .05) {return std::nullopt;}
      auto result = a;
      result.sample_monotonic = stamp;
      const double fraction = (stamp - a.sample_monotonic) / dt;
      for (size_t j = 0; j < result.q.size(); ++j) {
        result.q[j] += fraction * (b.q[j] - a.q[j]);
      }
      return result;
    }
    return std::nullopt;
  }

private:
  std::deque<LowStatePacket> samples_;
};
}  // namespace unitree_g1_kinematics
